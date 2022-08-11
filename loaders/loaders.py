from utils.miscs import  *
from NPP_proposal.feature_searching import *

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




def load_NPP_completion(args):
    '''
    load data information
    '''
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

