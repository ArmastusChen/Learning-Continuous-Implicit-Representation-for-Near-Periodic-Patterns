import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.activations import *
import kornia
# # TODO: remove this dependency
# from torchsearchsorted import searchsorted
import robust_loss_pytorch.general

# Misc
def img2mse(x, y, loss_type, adaptive, mask=None):
    if loss_type == 'robust_loss':
        loss = torch.mean(robust_loss_pytorch.general.lossfun(
            x - y, alpha=torch.Tensor([2.]), scale=torch.Tensor([0.1])))
    elif loss_type == 'l2':
        loss = (x - y) ** 2
    elif loss_type == 'robust_loss_adaptive':
        loss = torch.mean(adaptive.lossfun((x - y)))

    # if  mask is None:
    return torch.mean(loss)
    # else:
    #     return torch.mean(loss * mask * 20 + loss)

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)





