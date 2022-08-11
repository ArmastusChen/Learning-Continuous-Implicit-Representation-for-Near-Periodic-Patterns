import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from models.model_def import get_model_def
from utils.ops import PadMultipleOf
import torchvision.transforms as T
from utils.ops import gen_batches, calc_batch_size
import math
from utils.miscs import *
from utils.periodicity_visualizer import *


def im2act(im, mask, model_name='alexnet', gray_only=False):
    '''
        get activation map from rgb image
    '''
    model_def = get_model_def(model_name)
    model = model_def.get_model(use_gpu=True)
    image_transform = T.Compose([
        PadMultipleOf(32),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with model_def.hook_model(model) as extractor:
        if im.shape[-1] == 4:
            im = im[..., :3]

        image = Image.fromarray(im[:, :, :]).convert('RGB')
        image = image_transform(image).unsqueeze(0)
        image = image.cuda()
        activation = extractor(image)[0][0]


    image_shape = np.array(im.shape[:2])
    new_shape = image_shape // 4

    mask = torch.tensor(cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST), dtype=torch.float32).cuda()
    mask = mask[None]

    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = cv2.resize(im, (new_shape[1] * 2, new_shape[0]*2) )
    im = cv2.resize(im, (new_shape[1], new_shape[0]) )
    im = torch.Tensor(im[None]).cuda()

    if gray_only:
        activation = torch.cat([ im, mask], dim=0)
    else:
        activation = torch.cat([activation[:, :new_shape[0], :new_shape[1]], im, mask], dim=0)

    return activation * mask, mask


def act2edge(activation_, mask):
    '''
    Perform edge detection
    '''
    activation_ = normalize_to_uint8(tensor2np(activation_), channel_idx=(1,2))
    mask = tensor2np(mask[0])
    activation_edge = np.zeros((1, activation_.shape[1], activation_.shape[2]))
    for conv_id in range(activation_.shape[0]):

        conv_feat = activation_[conv_id]
        conv_feat = canny(conv_feat, mask)

        activation_edge += conv_feat / 255
    activation_edge = torch.Tensor(activation_edge).float()
    activation_edge = torch.cat([activation_edge, torch.Tensor(mask)[None]])
    return activation_edge







def feature_search(activation_, mask, repeat_range = (3, 6, 1), edge_searching=True):
    '''
        Args:
            img: input image
            activation_:  activation map
            mask:  unknown mask
            repeat_range: key hyperparameters for the periodicity detection method [1]. Its format is (start_range, end_range, step).
                          That being said, the first group of hyperparameter is [start_range, start_range+step].
                          In this case, the range of 2D displacement vector to be searched is [img_size / (start_range+step), img_size / start_range]
            edge_searching: True if the activation_ is processed by edge searching
    '''

    all_selected_angles, all_selected_periods, all_selected_shifts = [], [], []

    # for loop for different hyperparameters
    for i in range(repeat_range[0], repeat_range[1], repeat_range[2]):
        repeat_range_x, repeat_range_y = (i, i+repeat_range[2]), (i, i+repeat_range[2])

        # generate all the possible shifts (displacements) that are in the range
        possible_shifts = generate_possible_shifts(activation_.shape[1:], repeat_range_x, repeat_range_y, activation_.device)

        # if none of the shifts are generated
        if len(possible_shifts) == 0:
            continue

        # compute the loss for all the possible shifts based on brute-force searching
        losses = compute_loss(activation_, mask, possible_shifts, repeat_range_x, edge_searching = edge_searching)

        # generate the best periodicity based on the computed loss.
        selected_angles, selected_periods, selected_shift = generate_periodicity(losses, possible_shifts)

        if selected_angles is None or selected_periods is None or selected_shift is None:
            continue

        all_selected_angles.append(selected_angles)
        all_selected_periods.append(selected_periods)
        all_selected_shifts.append(selected_shift)

    return all_selected_angles, all_selected_periods, all_selected_shifts


def generate_periodicity(losses, possible_shifts):
    '''
    generate the best periodicity based on the computed loss.

    Args:
        losses: losses for possible displacement vectors
        possible_shifts: possible displacement vectors
     '''

    # sort the losses
    sorted_index = torch.argsort(losses)
    sorted_shifts = possible_shifts[sorted_index].type(torch.float32)

    # find the index of another displacement vector
    second_index = find_second_shift_by_angle(sorted_shifts)

    # if second displacement not found
    if second_index is None:
        return None, None, None

    selected_shift = [sorted_shifts[0], sorted_shifts[second_index]]

    '''
    Convert displacement vectors into period and orientations. See sec 2.1 in Supp for detailed derivations. 
    '''
    # NOTE: angle for first displacement vector is computed based on the second displacement vector
    selected_angles = [shifts2angle(selected_shift[1][None]), shifts2angle(selected_shift[0][None])]
    selected_periods = []
    for angle_idx in range(len(selected_angles)):
        # current shift
        this_shift = selected_shift[angle_idx]
        # another shift
        another_shift = selected_shift[(angle_idx + 1) % 2]
        # compute period
        period = shifts2period(this_shift, another_shift)
        selected_periods.append(period)

    return selected_angles, selected_periods, selected_shift


def search_periodicity_by_feat(img, mask, repeat_range = (2, 32, 5), edge_searching = False, gray_only = False, threshold = 10):
    '''
        Perform periodicity detection method [1] for multiple times based on repeated range.
        It searches in the form of displacement vectors.

        Args:
            img: input image with mask.
            mask: unknown mask.
            repeat_range: key hyperparameters for the periodicity detection method [1]. Its format is (start_range, end_range, step).
                          That being said, the first group of hyperparameter is [start_range, start_range+step].
                          In this case, the range of 2D displacement vector to be searched is [img_size / (start_range+step), img_size / start_range]
            edge_searching: if True, apply edge detection for feature map.
            gray_only: only use gray-scale image as feature
            threshold: threshold for edge detection

        [1] http://p3i.csail.mit.edu/
    '''

    # get the feature map
    activation_, mask = im2act(img, mask, gray_only=gray_only)

    # do edge detection
    if edge_searching:
        activation_edge = act2edge(activation_[:-1], mask)
        if not edge_searching:
            activation_edge[0] = activation_edge[0] > threshold

        # mask out the activation value that are not on the edge region
        activation_ = activation_ * activation_edge[[0]]

    '''
    Do feature searching given the processed feature map
    '''
    selected_angles, selected_periods, selected_shifts = feature_search(activation_, mask[0],
                                                                        repeat_range = repeat_range,
                                                                        edge_searching=edge_searching)

    '''
    scale the detected periodicity 
    '''
    # compute scale ratio
    ratio = np.round(img.shape[0] / activation_.shape[1])
    for i in range(len(selected_periods)):
        selected_periods[i] = [selected_periods[i][j] * ratio for j in range(len(selected_periods[i]))]
        selected_shifts[i] = [selected_shifts[i][j] * ratio for j in range(len(selected_shifts[i]))]

    return selected_angles, selected_periods, selected_shifts



def compute_loss(activation_, mask, possible_shifts, repeat_range, memory_use=4, edge_searching=True):
    '''
        Args:
             activation_: activation_with_mask gpu tensor shape: (layer_nr, h, w)
             possible_shifts: gpu tensor shape: (possible_shift_nr, 2)


        return: batch loss: shape (bs, 1)
    '''

    '''
    pad canvas
    '''
    act_c_, act_h, act_w = activation_.shape
    pad_x, pad_y = act_w // repeat_range[0] + 2, act_h // repeat_range[0] + 2
    activation_pad_ = torch.zeros((act_c_, act_h + pad_y, act_w + pad_x * 2),
                                  dtype=activation_.dtype, device=activation_.device)
    mask_pad_ = torch.zeros((1, act_h + pad_y, act_w + pad_x * 2),
                                  dtype=activation_.dtype, device=activation_.device)
    activation_pad_[:, :act_h, pad_x:pad_x + act_w] = activation_
    mask_pad_[:, :act_h, pad_x:pad_x + act_w] = mask

    '''
    generate canvas index
    '''
    y_index, x_index = torch.meshgrid([
        torch.arange(act_h, device=activation_.device),
        torch.arange(pad_x, pad_x + act_w, device=activation_.device)
    ])  # shape: (h, w)
    index = torch.stack([x_index, y_index], dim=2)  # shape: (h, w, 2)

    '''
    batch compute overlap
    '''
    possible_shifts = possible_shifts.unsqueeze(1).unsqueeze(1)  # shape: (nr, 1, 1, 2)
    losses = torch.zeros(possible_shifts.shape[0], device=activation_.device)
    if possible_shifts.shape[0] == 0:
        assert False
    batches = gen_batches(possible_shifts.shape[0], batch_size=calc_batch_size(memory_use, activation_.numel()))
    for batch_start, batch_end in batches:
        index_shift = possible_shifts[batch_start:batch_end] + index  # shape: (bs, h, w, 2)
        batch_activation_ = activation_pad_[:, index_shift[..., 1], index_shift[..., 0]].transpose(0, 1)

        if edge_searching:
            # for non-edge region, by doing multiplication, the difference score is 0.
            pow_diff = -batch_activation_[:, :-1] * activation_[:-1]
        else:
            diff = batch_activation_[:, :-1] - activation_[:-1]  # shape: (bs, layer_nr, h, w)
            pow_diff = torch.pow(diff, 2)  # shape: (bs, layer_nr, h, w)

        # mask after shifting
        shift_mask = mask_pad_[0, index_shift[..., 1], index_shift[..., 0]]

        # only valid when region before and after shifting are NOT in the unknown region
        losses[batch_start:batch_end] = torch.sum(pow_diff * mask[None, None, ...] * shift_mask[:, None, ...] , dim=[1, 2, 3])

    return losses


def generate_possible_shifts(act_shape, repeat_range_x=(1, 10), repeat_range_y =(10,20), device='cpu'):
    dxs, dys = torch.meshgrid([
        torch.arange(-act_shape[1] // repeat_range_x[0], act_shape[1] // repeat_range_x[0], device=device),
        torch.arange(0, act_shape[0] // repeat_range_y[0], device=device)
    ])
    possible_shifts = torch.stack([dxs.flatten(), dys.flatten()], dim=1).long()
    # ignore minor shift
    select = (torch.abs(possible_shifts[:, 0]) > act_shape[1] // repeat_range_x[1]) | \
             (possible_shifts[:, 1] > act_shape[0] // repeat_range_y[1])
    possible_shifts = possible_shifts[select]
    return possible_shifts



def find_second_shift_by_angle(sorted_shifts, minimum_angle=20):
    '''
    Find the second displacement vector

    Args:
        sorted_shifts: displacement vectors that are sorted based on the losses
        minimum_angle: the minimum angle difference between the first and second displacement vectors

    Returns:
        index of selected second displacement vector
    '''

    # get the angles between x-axis and all displacement vectors
    sorted_thetas = torch.atan2(sorted_shifts[:, 1], sorted_shifts[:, 0]) * 180 / math.pi
    # compute angle difference between all candidate displacement vectors and the first one
    sorted_angle = torch.abs(sorted_thetas - sorted_thetas[0])

    # filter those with small angle difference
    select = (sorted_angle > minimum_angle) & (sorted_angle < 180 - minimum_angle)
    select_indexes = select.nonzero()

    # if second direction founded
    if select_indexes.shape[0]:
        return select_indexes[0][0]

    return None


def shifts2angle(shifts):
    '''
    Convert displacement to angle
    '''
    angle = 180 - torch.atan2(shifts[:, 1], shifts[:, 0]) * 180 / math.pi
    return angle


def shifts2period(this_shift, another_shift):
    '''
    Convert displacement to period
    '''
    # length of vector
    period = torch.sqrt(this_shift[1] ** 2 + this_shift[0] ** 2)

    # angle difference between two vectors
    phi = angle_diff(this_shift, another_shift)
    period = period * torch.sin(phi)
    return period


def vector_norm(vector):
    """ Returns the unit vector of the vector.  """
    return vector / torch.linalg.norm(vector)

def angle_diff(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = vector_norm(v1)
    v2_u = vector_norm(v2)
    return torch.arccos(torch.clamp(torch.dot(v1_u, v2_u), -1.0, 1.0))