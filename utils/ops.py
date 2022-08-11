
class PadMultipleOf(object):
    def __init__(self, multiple, mode='constant', fill=0):
        self.multiple = multiple
        self.mode = mode
        self.fill = fill

    def __call__(self, img):
        return pad_multiple_of(img, self.multiple, mode=self.mode, fill=self.fill)
from PIL import Image
import numpy as np

from torchvision.transforms import functional as TF

___all__ = ['pad_multiple_of', 'PadMultipleOf']


def calc_batch_size(memory_use, unit_size):
    return int(memory_use * 1e9 / (unit_size * 32))


def pad(img, padding, mode='constant', fill=0):
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    else:
        assert len(padding) == 4

    if mode == 'constant':
        img_new = TF.pad(img, padding, fill=fill)
    else:
        np_padding = ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0))
        img_new = Image.fromarray(np.pad(
            np.array(img), np_padding, mode=mode
        ))

    return img_new


def pad_multiple_of(img, multiple, mode='constant', fill=0):
    h, w = img.height, img.width
    hh = h - h % multiple + multiple * int(h % multiple != 0)
    ww = w - w % multiple + multiple * int(w % multiple != 0)
    if h != hh or w != ww:
        return pad(img, (0, 0, ww - w, hh - h), mode=mode, fill=fill)
    return img




def gen_batches(nr, batch_size):
    batch_starts = np.arange(0, nr, batch_size)
    batch_ends = batch_starts + batch_size
    batch_ends[-1] = nr
    return np.stack([batch_starts, batch_ends], axis=1)


