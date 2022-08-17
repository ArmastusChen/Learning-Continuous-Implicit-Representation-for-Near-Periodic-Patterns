import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from externel_lib.robust_loss_pytorch import AdaptiveLossFunction
import numpy as np
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, use_adaptive=False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        self.use_adaptive = use_adaptive

        if use_adaptive:
            self.adaptives = []
            chns = [64, 128, 256]
            for chn in chns:
                self.adaptives.append(AdaptiveLossFunction(
                num_dims=chn ** 2, float_dtype=np.float32, device=0))

            # fix the encoder
            for i in range(3):
                for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                    param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def style_loss(self, A_img, B_img, weight=None):
        A_feats = self(A_img)
        B_feats = self(B_img)

        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            # feat_mask = F.interpolate(mask, size=(w, h), mode='nearest')
            # A_feat = A_feat * feat_mask
            # B_feat = B_feat * feat_mask
            # plt.imshow(A_feat.detach().cpu()[0,0])
            # plt.show()
            # plt.imshow(B_feat.detach().cpu()[0,0])
            # plt.show()
            # plt.imshow(feat_mask.detach().cpu()[0,0])
            # plt.show()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            if not self.use_adaptive:
                loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
            else:
                A_style = A_style.reshape(A_style.shape[0], -1)
                B_style = B_style.reshape(B_style.shape[0], -1)
                # loss = (self.adaptives[i].lossfun((A_style - B_style)))
                if weight is None:
                    loss_value += torch.mean((self.adaptives[i].lossfun((A_style - B_style))) / (c * w * h))
                else:
                    tmp_loss = torch.mean((self.adaptives[i].lossfun((A_style - B_style))) / (c * w * h), dim=-1)
                    tmp_loss = tmp_loss * weight
                    loss_value += torch.sum(tmp_loss)


        return loss_value

