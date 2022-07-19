import torch
import torch.nn as nn

from typing import Union, Tuple


def extract_glimpse(input: torch.Tensor, size: Tuple[int, int], offsets, centered=True, normalized=True, mode='bilinear', padding_mode='zeros'):
    '''Returns a set of windows called glimpses extracted at location offsets
    from the input tensor. If the windows only partially overlaps the inputs,
    the non-overlapping areas are handled as defined by :attr:`padding_mode`.
    Options of :attr:`padding_mode` refers to `torch.grid_sample`'s document.

    The result is a 4-D tensor of shape [N, C, h, w].  The channels and batch
    dimensions are the same as that of the input tensor.  The height and width
    of the output windows are specified in the size parameter.

    The argument normalized and centered controls how the windows are built:

        * If the coordinates are normalized but not centered, 0.0 and 1.0 correspond
          to the minimum and maximum of each height and width dimension.
        * If the coordinates are both normalized and centered, they range from
          -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper left
          corner, the lower right corner is located at (1.0, 1.0) and the center
          is at (0, 0).
        * If the coordinates are not normalized they are interpreted as numbers
          of pixels.

    Args:
        input (Tensor): A Tensor of type float32. A 4-D float tensor of shape
            [N, C, H, W].
        size (tuple): 2-element integer tuple specified the
            output glimpses' size. The glimpse width must be specified first,
            following by the glimpse height.
        offsets (Tensor): A Tensor of type float32. A 2-D integer tensor of
            shape [batch_size, 2]  containing the x, y locations of the center
            of each window.
        centered (bool, optional): An optional bool. Defaults to True. indicates
            if the offset coordinates are centered relative to the image, in
            which case the (0, 0) offset is relative to the center of the input
            images. If false, the (0,0) offset corresponds to the upper left
            corner of the input images.
        normalized (bool, optional): An optional bool. Defaults to True. indicates
            if the offset coordinates are normalized.
        mode (str, optional): Interpolation mode to calculate output values.
            Defaults to 'bilinear'.
        padding_mode (str, optional): padding mode for values outside the input.
    Raises:
        ValueError: When normlised set False but centered set True

    Returns:
        output (Tensor): A Tensor of same type with input.
    '''
    W, H = input.size(-1), input.size(-2)

    if normalized and centered:
        offsets = (offsets + 1) * offsets.new_tensor([W/2, H/2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W, H])
    elif centered:
        raise ValueError(
            f'Invalid parameter that offsets centered but not normlized')

    h, w = size
    xs = torch.arange(0, w, dtype=input.dtype,
                      device=input.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=input.dtype,
                      device=input.device) - (h - 1) / 2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W/2, H/2])) / offsets_grid.new_tensor([W/2, H/2])

    return torch.nn.functional.grid_sample(
        input, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode)


def extract_multiple_glimpse(input: torch.Tensor, size: Tuple[int, int], offsets, centered=True, normalized=True, mode='bilinear'):
    # offsets: [B, n, 2]
    patches = []
    for i in range(offsets.size(-2)):
        patch = extract_glimpse(
            input, size, offsets[:, i, :], centered, normalized, mode)
        patches.append(patch)
    return torch.stack(patches, dim=1)
