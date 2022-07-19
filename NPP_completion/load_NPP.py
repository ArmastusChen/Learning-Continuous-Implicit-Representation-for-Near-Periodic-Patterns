import os

import numpy as np
import json
import cv2


def load_NPP_data(args):

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
    if args.extrapolation:
        valid_mask = np.ones_like(valid_mask)
    '''
    get pixel coordinate of training (known) and val (unknown)
    '''
    train_splits = np.stack(np.nonzero(mask * valid_mask)[:2], axis=1)
    val_splits = np.stack(np.nonzero((1-mask) * valid_mask)[:2], axis=1)
    i_split = [ train_splits, val_splits]

    if args.normalize_type == 2:
        img = (img - 0.5) * 2

    img = img[None]
    mask = mask[None]
    valid_mask = valid_mask[None]
    masked_img = masked_img[None]

    '''
    load detected periodicity
    '''
    selected_shifts, selected_angles, selected_periods = data_info['selected_shifts'], data_info['selected_angles'], \
                                                         data_info['selected_periods']

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
