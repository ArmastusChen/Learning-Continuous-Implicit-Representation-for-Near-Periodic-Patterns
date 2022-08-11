import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from .activations import *


# Model
class NPP_Net(nn.Module):
    def __init__(self, input_ch_periodic, input_ch_periodic_aux, freq_scales, freq_offsets, angle_offsets, D=8, W=256, freq_nerf=3, output_ch=3, skips=[4], activation='relu'):
        """
        Args:
            input_ch_periodic:  input channel of periodic positional encoding of top-1 periodicity (before applying nerf positional encoding)
            input_ch_periodic_aux:  input channel of periodic positional encoding of top-2 to K periodicity (before applying nerf positional encoding)
            freq_scales: a set of fine level periodicity augmentation: augmented_p = freq_scale * p
            freq_offsets: a set of fine level periodicity augmentation: augmented_p = freq_offset + p
            angle_offsets: a set of fine level periodicity augmentation: augmented_orientation = orientation + angle_offset
            freq_nerf: the dimension of original nerf positional encoding

            The rest of args:  network parameters
        """
        super(NPP_Net, self).__init__()
        self.scale = len(freq_scales)
        self.offset = len(freq_offsets)
        self.angle_offset = len(angle_offsets)

        #  Dimension of Top-1 positional encoding (after applying nerf positional encoding)
        input_ch_periodic = input_ch_periodic * freq_nerf
        self.input_ch_periodic = input_ch_periodic

        #  Dimension of Top-2 to K positional encoding (after applying nerf positional encoding)
        input_ch_periodic_aux = input_ch_periodic_aux * freq_nerf
        self.input_ch_periodic_aux = input_ch_periodic_aux


        '''
        Network settings
        '''
        self.D = D
        self.W = W
        self.skips = skips

        self.periodic_linears = nn.ModuleList(
            [nn.Linear(input_ch_periodic, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_periodic, W) for i in range(D-1)])
        self.scale_linears = nn.ModuleList([nn.Linear(input_ch_periodic_aux + W, W)])
        self.pos_linears = nn.ModuleList([nn.Linear(( W + W ) , W//2)])
        self.feature_linear1 = nn.Linear(W, W)
        self.feature_linear2 = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, output_ch)

        if activation == 'snake':
            self.snakes = SnakeActivation()
        else:
            self.snakes = None

    def forward(self, x, x_periodic):
        # top1
        input_periodic = x_periodic[:, :self.input_ch_periodic]
        # top 2 to K
        input_scale_periodic = x_periodic[:, self.input_ch_periodic:]
        assert input_scale_periodic.shape[1] == self.input_ch_periodic_aux

        h = input_periodic
        for i, l in enumerate(self.periodic_linears):
            h = self.periodic_linears[i](h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)
            if i in self.skips:
                h = torch.cat([input_periodic, h], -1)

        feature1 = self.feature_linear1(h)

        # Top-2 to K MLP
        h = torch.cat([feature1, input_scale_periodic], -1)
        for i, l in enumerate(self.scale_linears):
            h = self.scale_linears[i](h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)

        feature2 = self.feature_linear2(h)
        h = torch.cat([feature1, feature2], -1)

        for i, l in enumerate(self.pos_linears):
            h = self.pos_linears[i](h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)

        outputs = self.rgb_linear(h)
        return outputs


# Model
class NPP_Net_top1(nn.Module):
    def __init__(self, input_ch_periodic, freq_scales, freq_offsets, angle_offsets, D=8, W=256,
                 freq_nerf=3, output_ch=3, skips=[4], activation='relu'):
        """
        Args:
            input_ch_periodic:  input channel of periodic positional encoding of top-1 periodicity (before applying nerf positional encoding)
            input_ch_periodic_aux:  input channel of periodic positional encoding of top-2 to K periodicity (before applying nerf positional encoding)
            freq_scales: a set of fine level periodicity augmentation: augmented_p = freq_scale * p
            freq_offsets: a set of fine level periodicity augmentation: augmented_p = freq_offset + p
            angle_offsets: a set of fine level periodicity augmentation: augmented_orientation = orientation + angle_offset
            freq_nerf: the dimension of original nerf positional encoding

            The rest of args:  network parameters
        """
        super(NPP_Net_top1, self).__init__()
        self.scale = len(freq_scales)
        self.offset = len(freq_offsets)
        self.angle_offset = len(angle_offsets)
        self.scale_dim = (self.scale - 1) * freq_nerf * (
            self.offset) * self.angle_offset * 2  # 2 refers to two orientation

        input_ch_periodic = input_ch_periodic * freq_nerf
        self.input_ch_periodic = input_ch_periodic

        self.D = D
        self.W = W
        self.skips = skips

        self.periodic_linears = nn.ModuleList(
            [nn.Linear(input_ch_periodic, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_periodic, W) for i in range(D - 1)])


        self.pos_linears = nn.ModuleList([nn.Linear((W), W // 2)])

        self.feature_linear1 = nn.Linear(W, W)
        self.feature_linear2 = nn.Linear(W, W)

        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, output_ch)

        if activation == 'snake':
            self.snakes = SnakeActivation()
        else:
            self.snakes = None

    def forward(self, x, x_periodic):

        input_periodic = x_periodic[:, :self.input_ch_periodic]
        assert x_periodic.shape[1] == self.input_ch_periodic

        h = input_periodic
        for i, l in enumerate(self.periodic_linears):
            h = self.periodic_linears[i](h)
            # h = F.relu(h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)
            if i in self.skips:
                h = torch.cat([input_periodic, h], -1)

        feature1 = self.feature_linear1(h)

        h = torch.cat([feature1], -1)

        for i, l in enumerate(self.pos_linears):
            h = self.pos_linears[i](h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)

        outputs = self.rgb_linear(h)
        return outputs


class NPP_Net_light(nn.Module):
    def __init__(self, input_ch_periodic, freq_scales, freq_offsets, angle_offsets, D=8, W=256, input_ch=3, output_ch=3, skips=[4], activation='relu'):
        """
        """
        super(NPP_Net_light, self).__init__()
        self.scale = len(freq_scales)
        self.offset = len(freq_offsets)
        self.angle_offset = len(angle_offsets)
        self.scale_dim = (self.scale - 1) * 4 * (self.offset) * self.angle_offset

        input_ch_periodic_all = input_ch_periodic
        input_ch_periodic = 2 * ( 2 * self.offset * self.angle_offset)

        self.scale_inds = [i for i in range( 2 * self.offset * self.angle_offset,  2 * self.offset * self.angle_offset + self.scale_dim // 2)] + [i for i in range(input_ch_periodic_all - self.scale_dim // 2, input_ch_periodic_all)]
        self.period_inds = [i for i in range(input_ch_periodic_all) if i not in self.scale_inds]

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_periodic = input_ch_periodic
        self.skips = skips

        self.periodic_linears = nn.ModuleList(
            [nn.Linear(input_ch_periodic, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_periodic, W) for i in range(D-1)])

        self.scale_linears =nn.ModuleList([nn.Linear(self.scale_dim + W, W)])


        self.pos_linears = nn.ModuleList([nn.Linear((input_ch + W + W )if self.scale > 1 else (input_ch + W) , W//2)])


        self.feature_linear1 = nn.Linear(W, W)
        self.feature_linear2 = nn.Linear(W, W)


        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, output_ch)

        if activation == 'snake':
            self.snakes = SnakeActivation()
        else:
            self.snakes = None

    def forward(self, x, x_periodic):
        input_pos = x
        input_periodic = x_periodic[:, self.period_inds]
        input_scale_periodic = x_periodic[:, self.scale_inds]

        h = input_periodic
        for i, l in enumerate(self.periodic_linears):
            h = self.periodic_linears[i](h)
            # h = F.relu(h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)
            if i in self.skips:
                h = torch.cat([input_periodic, h], -1)

        feature1 = self.feature_linear1(h)

        if not self.scale == 1:
            # scale MLP
            h = torch.cat([feature1, input_scale_periodic], -1)
            for i, l in enumerate(self.scale_linears):
                h = self.scale_linears[i](h)
                if self.snakes == None:
                    h = F.relu(h)
                else:
                    h = self.snakes(h)

            feature2 = self.feature_linear2(h)
            h = torch.cat([feature1, feature2, input_pos], -1)
        else:
            h = torch.cat([feature1, input_pos], -1)

        # high freq MLP

        for i, l in enumerate(self.pos_linears):
            h = self.pos_linears[i](h)
            if self.snakes == None:
                h = F.relu(h)
            else:
                h = self.snakes(h)


        outputs = self.rgb_linear(h)
        return outputs




