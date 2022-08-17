import os

import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
from models.activations import *
# # TODO: remove this dependency
# from torchsearchsorted import searchsorted
print(os.getcwd())
import robust_loss_pytorch.general

# Misc
def img2mse(x, y, loss_type, adaptive, mask=None):
    diff = x - y

    if mask is not None:
        diff = diff * mask + (1-mask) * diff * 0.3

    if loss_type == 'robust_loss':
        loss = torch.mean(robust_loss_pytorch.general.lossfun(
            diff, alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))
    elif loss_type == 'l2':
        loss = diff ** 2
    elif loss_type == 'robust_loss_adaptive':
        loss = torch.mean(adaptive.lossfun(diff))

    return torch.mean(loss)

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)





