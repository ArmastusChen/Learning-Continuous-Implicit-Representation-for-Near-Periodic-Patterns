import os

import matplotlib.pyplot as plt

from NPP_proposal.feature_searching import *
import json
from utils.ops import  *

def load_NPP_proposal(args):
    file_dir = args.datadir

    '''
    read images
    '''
    masked_img = cv2.imread(f'./{file_dir}/masked_img.png')[..., ::-1]
    img = cv2.imread(f'./{file_dir}/gt_img.png')[..., ::-1]
    mask = cv2.imread(f'./{file_dir}/unknown_mask.png', 0)[..., None]
    valid_mask = cv2.imread(f'./{file_dir}/valid_mask.png', 0)[..., None]

    mask = mask / 255
    valid_mask = valid_mask / 255
    masked_img = masked_img / 255.
    img = img / 255.

    '''
    search multiple periodicity given periodicity range
    '''
    selected_angles, selected_periods, selected_shifts = search_periodicity_by_feat(np.uint8(masked_img * 255),
                                                                                    np.uint8(valid_mask * mask)[..., 0],
                                                                                     repeat_range=args.search_range,
                                                                                    edge_searching=args.edge_searching,
                                                                                    gray_only=args.gray_only)

    '''
    Generate pseudo mask
    '''
    # find the top-K centroid of pseudo mask
    centroids, dist_to_mask = find_mask_centroid(mask * valid_mask)

    # generate pseudo mask
    pseudo_mask = np.ones_like(mask)
    for i in range(len(centroids)):
        centroid = centroids[i]
        # half size of the pseudo mask
        half_win = int(dist_to_mask[i] / np.sqrt(2) / 1.2)

        pseudo_mask[centroid[0]-half_win: centroid[0]+half_win, centroid[1]-half_win: centroid[1]+half_win, :] = 0

    '''
    get pixel coordinate of training (known) and val (unknown)
    '''
    train_splits = np.stack(np.nonzero(pseudo_mask * mask * valid_mask)[:2], axis=1)
    val_splits = np.stack(np.nonzero((1-pseudo_mask) * mask * valid_mask)[:2], axis=1)
    i_split = [train_splits, val_splits]

    if args.normalize_type == 2:
        img = (img - 0.5) * 2

    img = img[None]
    pseudo_mask = pseudo_mask[None]
    valid_mask = valid_mask[None]
    masked_img = masked_img[None]

    return img, pseudo_mask, mask, masked_img, valid_mask, i_split, selected_shifts, selected_angles, selected_periods


def load_data(args):
    data_info_tmp = [json.loads(x.rstrip()) for x in open(f'{args.datadir}/config.odgt', 'r')][0]
    data_info = {}
    for key in data_info_tmp:
        if 'fpath' in key:
            if isinstance(data_info_tmp[key], list):
                f_name = data_info_tmp[key][0].split('/')[-1]
            else:
                f_name = data_info_tmp[key].split('/')[-1]
            data_info[key] = f'{args.datadir}/{f_name}'
        else:
            data_info[key] = data_info_tmp[key]

    return data_info

def load_NPP_completion(args):
    '''
    load data information
    '''
    data_info = load_data(args)

    '''
    read images
    '''
    masked_img = cv2.imread(data_info['fpath_masked_img'])[:, :, ::-1]
    img = cv2.imread(data_info['fpath_gt_img'])[:, :, ::-1]
    valid_mask = cv2.imread(data_info['fpath_valid_mask'], 0)[:, :, None]
    mask = cv2.imread(data_info['fpath_mask'], 0)[:, :, None]
    img = img / 255.
    masked_img = masked_img / 255.
    valid_mask = valid_mask / 255.
    mask = mask / 255.

    # filter invalid region
    mask = mask * valid_mask
    if args.invalid_as_unknown:
        valid_mask = np.ones_like(valid_mask)
    '''
    get pixel coordinate of training (known) and val (unknown)
    '''
    train_splits = np.stack(np.nonzero(mask * valid_mask)[:2], axis=1)
    val_splits = np.stack(np.nonzero((1 - mask) * valid_mask)[:2], axis=1)
    i_split = [train_splits, val_splits]

    if args.normalize_type == 2:
        img = (img - 0.5) * 2

    img = img[None]
    mask = mask[None]
    valid_mask = valid_mask[None]
    masked_img = masked_img[None]

    '''
    load detected periodicity
    '''
    selected_shifts, selected_angles, selected_periods = \
        data_info['selected_shifts'], data_info['selected_angles'], data_info['selected_periods']

    # use top K periodicity
    selected_shifts = selected_shifts[:args.p_topk]
    selected_angles = selected_angles[:args.p_topk]
    selected_periods = selected_periods[:args.p_topk]

    '''
    calculate patch size 
    '''
    max_period = max(selected_periods[0])
    args.patch_size = int(np.clip(max_period + (32 - max_period % 32), a_min=64, a_max=160))

    return img, mask, masked_img, valid_mask, i_split, selected_shifts, selected_angles, selected_periods




def load_NPP_segmentation(args):
    import NPP_segmentation.imsegm.pipelines as segm_pipe

    '''
    load data information
    '''
    data_info = load_data(args)

    '''
    read images
    '''
    img = cv2.imread(data_info['fpath_gt_img'])[:, :, ::-1]
    valid_mask = cv2.imread(data_info['fpath_valid_mask'], 0)

    valid_mask = valid_mask / 255.

    # we use blur image for training because we don't want to consider local details
    blur_img = blur_with_mask(img, valid_mask[..., None] )
    blur_img = blur_img / 255.


    '''
    Do the coarse semantic segmentation for initial periodic region estimation.
    '''
    # valid mask
    mask_for_seg = valid_mask > 0.5
    nb_classes, sp_size, sp_regul = args.nb_classes, args.sp_size, args.sp_regul

    # estimate a model from the image and return it as result
    dict_features = {'color': ['mean', 'median', 'meanGrad']}
    model, _ = segm_pipe.estim_model_classes_group([img], nb_classes, sp_size=sp_size, sp_regul=sp_regul, mask=mask_for_seg,
                                                   dict_features=dict_features, pca_coef=None, model_type='GMM')

    # complete pipe-line for segmentation using superpixels, extracting features and graphCut segmentation
    dict_debug = {}
    seg, _ = segm_pipe.segment_color2d_slic_features_model_graphcut(img, model, mask=mask_for_seg, sp_size=sp_size,
                                                                    sp_regul=sp_regul,
                                                                    dict_features=dict_features, gc_regul=2,
                                                                    gc_edge_type='features', debug_visual=dict_debug)

    # increase segment label by 1, and 0 refers to invalid region now
    seg = (seg + 1) * valid_mask
    seg = np.uint8(seg)


    '''
    Generate non-periodic and periodic region based on initial coarse segmentation
    '''
    h, w = seg.shape
    # crop the segmentation result, and treat the label with largest number of pixels as periodic class
    period_label = np.bincount(seg[h//4:h//4 * 3, w//4: w//4 * 3].reshape(-1))[1:].argmax() + 1

    # other labels as non-periodic region
    other_labels = []
    for label_idx in range(1, nb_classes + 1):
        if not label_idx == period_label:
            other_labels.append(label_idx)

    # mask of non-periodic region
    non_period_mask = np.zeros_like(valid_mask[..., None])
    for label_idx in other_labels:
        non_period_mask[seg == label_idx] += 1

    # mask of periodic region
    period_mask = seg == period_label

    # visualize the initial segmentation
    name = args.datadir.split('/')[-1]
    expname = f'{args.expname}_top{args.p_topk}'
    savedir = f'{args.basedir}/{expname}/{name}/segment_init.png'
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    cv2.imwrite(savedir, np.uint8((non_period_mask > 0).astype(np.float) * 255))

    img = img / 255.

    img = img[None]
    blur_img = blur_img[None]
    period_mask = period_mask[None, ..., None]
    non_period_mask = non_period_mask[None]
    valid_mask = valid_mask[None, ..., None]

    '''
    load detected periodicity
    '''
    selected_shifts, selected_angles, selected_periods = data_info['selected_shifts'],  data_info['selected_angles'],  data_info['selected_periods']

    # use top K periodicity
    selected_shifts = selected_shifts[:args.p_topk]
    selected_angles = selected_angles[:args.p_topk]
    selected_periods = selected_periods[:args.p_topk]

    '''
    calculate patch size 
    '''
    max_period = max(selected_periods[0])
    args.patch_size = int(np.clip(max_period + (32 - max_period % 32), a_min=64, a_max=160))


    return img, period_mask, non_period_mask, blur_img, valid_mask, selected_shifts, selected_angles, selected_periods




def load_NPP_remapping(args):
    import NPP_remapping.blur_detection as blur_detection

    '''
    load data information
    '''
    data_info = load_data(args)

    '''
     read images
    '''
    img = cv2.imread(data_info['fpath_gt_img'])[:, :, ::-1]
    valid_mask = cv2.imread(data_info['fpath_valid_mask'], 0)[:, :, None]

    '''
    detect the blurry region in the image 
    '''
    _, clear_mask = blur_detection.get_blur_map(img, thresh=args.blur_thresh)

    clear_mask = clear_mask[:, :, None] * valid_mask / 255

    # visualization
    name = args.datadir.split('/')[-1]
    expname = f'{args.expname}_top{args.p_topk}'
    savedir = f'{args.basedir}/{expname}/{name}/blur_mask.png'
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    plt.imsave(savedir, clear_mask[..., 0])

    img = img / 255.
    valid_mask = valid_mask / 255.
    clear_mask = clear_mask / 255.

    '''
    get pixel coordinate of training (known) and val (unknown)
    '''
    train_splits = np.stack(np.nonzero(valid_mask)[:2], axis=1)
    val_splits = np.stack(np.nonzero(clear_mask * valid_mask)[:2], axis=1)
    i_split = [train_splits, val_splits]

    img = img[None]
    clear_mask = clear_mask[None]
    valid_mask = valid_mask[None]

    '''
    load detected periodicity
    '''
    selected_shifts, selected_angles, selected_periods = \
        data_info['selected_shifts'],  data_info['selected_angles'],  data_info['selected_periods']
    # use top K periodicity
    selected_shifts = selected_shifts[:args.p_topk]
    selected_angles = selected_angles[:args.p_topk]
    selected_periods = selected_periods[:args.p_topk]

    '''
     calculate patch size 
     '''
    max_period = max(selected_periods[0])
    args.patch_size = int(np.clip(max_period + (32 - max_period % 32), a_min=64, a_max=160))


    return img, clear_mask, valid_mask, i_split, selected_shifts, selected_angles, selected_periods