

from PIL import Image
import numpy as np

from torchvision.transforms import functional as TF
import cv2

___all__ = ['pad_multiple_of', 'PadMultipleOf']


class PadMultipleOf(object):
    def __init__(self, multiple, mode='constant', fill=0):
        self.multiple = multiple
        self.mode = mode
        self.fill = fill

    def __call__(self, img):
        return pad_multiple_of(img, self.multiple, mode=self.mode, fill=self.fill)


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




def blur_with_mask(img, mask, sigma=3, multichannel=True):
    from skimage.filters import gaussian

    img1 = gaussian(img * mask, sigma=sigma, multichannel=multichannel)

    img2 = gaussian(mask, sigma=sigma, multichannel=multichannel)

    blur_img = img1 / (img2+1e-6)
    blur_img = blur_img * mask

    return blur_img





def find_contours(img):
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]



def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255