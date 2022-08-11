import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import scipy.ndimage as ndimage
import scipy.cluster.hierarchy as hcluster
from skimage.feature import peak_local_max

from scipy.ndimage.filters import maximum_filter, gaussian_filter


def mask2ltrb(mask: torch.Tensor):
    fg_coord = mask.nonzero()
    brtl = torch.cat([fg_coord.max(dim=0)[0], fg_coord.min(dim=0)[0]])
    return brtl[[3, 2, 1, 0]]

def canny(img, mask):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    mask = ndimage.binary_erosion(mask, iterations=4).astype(np.float)
    img_canny = cv2.Canny(img_blur, 10, 100)
    img_canny = img_canny * mask

    return img_canny




def tensor2np(tensor):
    return tensor.cpu().numpy()


def normalize_to_uint8(array, channel_idx=-1):
    array_max = np.max(array, axis=channel_idx, keepdims=True)
    array_min = np.min(array, axis=channel_idx, keepdims=True)

    array = (array - array_min) / (array_max - array_min)

    return np.uint8(array * 255)




def find_mask_centroid(mask, topk=3, threshold_ratio=0.3):
    '''
    Find the pixels that are far away from the image boundary and unknown regions

    Args：
        mask: unknown mask for centroid selection
        topk: the number (top-K) of centroid to be selected
        threshold_ratio: threshold for the distance between selected centroid (should not be too small)

    Returns:
        centroids: coordinate of centroids
        selected_dis: distance from centriods to the boundary
    '''

    # distance of every pixel to borders
    dis = ndimage.distance_transform_edt(mask)
    dis = dis.reshape(-1)
    sorted_idx = np.argsort(-dis)
    # threshold for the distance between two centroid
    threshold = min(mask.shape[0], mask.shape[1]) * threshold_ratio

    centroids = []
    selected_dis = []
    for idx in sorted_idx:
        # height and width of current centroid
        h, w = idx // mask.shape[1], idx % mask.shape[1]

        # Based on the added centroids, check whether to add the current centroid
        isadded = True
        for centroid in centroids:
            # distance between two centroids
            related_dis = np.sqrt((centroid[0] - h) ** 2 + (centroid[1] - w) ** 2)

            if related_dis < threshold:
                isadded = False

        if isadded:
            centroids.append([h, w])
            selected_dis.append(dis[idx])

        # if top-K are already selected
        if len(selected_dis) == topk:
            break

    return centroids, selected_dis



