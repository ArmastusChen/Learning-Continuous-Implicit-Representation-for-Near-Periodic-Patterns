import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
class GridProgram(object):
    def __init__(self, resolution=None, base_point=None, first_shift=None, second_shift=None, loss=None,
                 grid_prog=None):
        if grid_prog is not None:
            self.resolution = grid_prog.resolution
            self.base_point = grid_prog.base_point
            self.first_shift = grid_prog.first_shift
            self.second_shift = grid_prog.second_shift
            self.loss = grid_prog.loss
        else:
            self.resolution = resolution
            self.base_point = base_point.cpu().numpy()
            self.first_shift = first_shift.cpu().numpy()
            self.second_shift = second_shift.cpu().numpy()
            self.loss = loss.cpu().numpy()

    def calc_resize_ratio(self, old_shape, new_shape):
        return np.array([new_shape[1] / old_shape[1], new_shape[0] / old_shape[0]], dtype=np.float32)

    def fit_resolution(self, target_resolution):
        resize_ratio = self.calc_resize_ratio(self.resolution, target_resolution)
        self.base_point = np.round(self.base_point * resize_ratio).astype(np.int32)
        self.first_shift *= resize_ratio
        self.second_shift *= resize_ratio

    def gen_ij(self, canvas_shape):
        CANVAS_CORNER = [[0, 0], [0, 1], [1, 0], [1, 1]]
        vectors = np.array(CANVAS_CORNER) * np.array(canvas_shape[::-1]) - self.base_point
        canvas_corner_coord = np.linalg.inv(np.stack([self.first_shift, self.second_shift], axis=1)) @ vectors.T
        # first: i; second: j;
        i_min, j_min = np.floor(canvas_corner_coord.min(axis=1)).astype(np.int)
        i_max, j_max = np.ceil(canvas_corner_coord.max(axis=1)).astype(np.int)
        return i_min, i_max, j_min, j_max

    def draw(self, image, color=(255, 255, 0), thickness=2):
        # resize the image accordingly
        self.fit_resolution(image.shape[:2])
        canvas = image[:, :, :-1].copy()

        i_min, i_max, j_min, j_max = self.gen_ij(canvas.shape[:2])

        i_base_points = self.base_point + np.arange(i_min, i_max)[..., np.newaxis] * self.first_shift
        i_lines = np.concatenate((i_base_points, i_base_points), axis=1)
        i_lines[:, :2] += j_min * self.second_shift
        i_lines[:, 2:] += j_max * self.second_shift

        j_base_points = self.base_point + np.arange(j_min, j_max)[..., np.newaxis] * self.second_shift
        j_lines = np.concatenate((j_base_points, j_base_points), axis=1)
        j_lines[:, :2] += i_min * self.first_shift
        j_lines[:, 2:] += i_max * self.first_shift

        lines = np.round(np.concatenate((i_lines, j_lines))).astype(np.int32)
        canvas_mask = np.zeros_like(canvas[..., 0 ])

        for line in lines:
            cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), color=color, thickness=thickness)
            line_img = np.zeros_like(canvas[..., 0 ])
            cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color=(1), thickness=thickness)

            canvas_mask += line_img

        return np.concatenate([canvas, image[:, :, -1:]], axis=2), canvas_mask

    def __str__(self):
        return f"base point: {self.base_point}\n" \
               f"first shift: {self.first_shift}\n" \
               f"second shift: {self.second_shift}\n" \
               f"loss: {self.loss:.2e}"