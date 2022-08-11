import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np

class Embedder:
    def __init__(self, res, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn(res)

    def create_embedding_fn(self, res):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']


        if self.kwargs['sampling'] == 'log':
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        elif self.kwargs['sampling'] == 'gaussian':
            freq_bands = torch.normal(mean=0.0, std=1.0, size=(N_freqs, 1)) * 10

            # freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        elif self.kwargs['sampling'] == 'pure_gaussian':

            dic1 = torch.zeros((1, int(N_freqs / 2)))
            for i in range(int(N_freqs / 2)):
                dic1[0, i] = 2 * i / N_freqs

            dic2 = torch.zeros((1, int(N_freqs / 2)))
            for i in range(int(N_freqs / 2)):
                dic2[0, i] = 2 * i / N_freqs
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        self.res = res
        self.is_search = self.kwargs['is_search']

    def embed(self, inputs):
        if self.is_search:
            inputs[:, 0] = ((inputs[:, 0] / self.res[0]) - 0.5) * 2
            inputs[:, 1] = ((inputs[:, 1] / self.res[1]) - 0.5) * 2

        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)



def get_embedder(multires, i=0, res=None, selected_angles=None, selected_periods=None,
                 freq_scales=None, freq_offsets=None, angle_offsets=None, is_search=False):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'sampling' : 'gaussian',
                'periodic_fns' : [torch.sin, torch.cos],
                'is_search': is_search,
    }

    # create positional encoder like original nerf
    if selected_periods is None and selected_angles is None:
        if is_search:
            embed_kwargs['input_dims'] = 2
        else:
            embed_kwargs['input_dims'] = 1

        embedder = Embedder(res, **embed_kwargs)
    # create positional encoder using periodicity
    else:
        if is_search:
            embed_kwargs['include_input'] = False

        embedder = Embedder_periodic(res, selected_angles, selected_periods, freq_scales, freq_offsets, angle_offsets, **embed_kwargs)

    return embedder, embedder.out_dim



class Embedder_periodic:
    def __init__(self, res, selected_angles, selected_periods, freq_scales, freq_offsets, angle_offsets, **kwargs):
        self.kwargs = kwargs
        self.freq_scales = freq_scales
        self.freq_offsets = freq_offsets
        self.angle_offsets =angle_offsets
        self.create_embedding_fn(res, selected_angles, selected_periods)

    def create_embedding_fn(self, res, selected_angles, selected_periods):
        embed_fns_x = []
        embed_fns_y = []

        out_dim = 0

        freq_bands = torch.Tensor(selected_periods)

        if self.kwargs['include_input']:
            # add original coordinates and normalize it to [-1, 1]
            embed_fns_x.append(lambda x, y: (x / res[1] - 0.5) * 2)
            embed_fns_y.append(lambda x, y: (y / res[0] - 0.5) * 2)
            out_dim += 2

        # periodicity-aware input warping (Eq. 1 in the main paper)
        for freq_scale in self.freq_scales:
            for freq_offset in self.freq_offsets:
                for idx in range(len(selected_angles)):
                    for angle_offset in self.angle_offsets:
                        freq = (freq_bands[idx] + freq_offset) * freq_scale

                        selected_angle = selected_angles[idx] + angle_offset
                        theta_r = torch.deg2rad(selected_angle)

                        for p_fn in self.kwargs['periodic_fns']:
                            embed_fns = lambda x, y, p_fn=p_fn,freq=freq, theta_r=theta_r: p_fn((  ( (y * torch.cos(theta_r) + x * torch.sin(theta_r)) % freq ) / freq  ) * 2 * np.pi)

                            if idx == 0:
                                embed_fns_x.append(embed_fns)
                            else:
                                embed_fns_y.append(embed_fns)
                            out_dim += 1


        self.embed_fns_x = embed_fns_x
        self.embed_fns_y = embed_fns_y
        self.out_dim = out_dim

    def embed(self, inputs):
        inputs_x = inputs[:, 1:2]
        inputs_y = inputs[:, 0:1]

        fn_x = torch.cat([fn(inputs_x, inputs_y) for fn in self.embed_fns_x], -1)
        fn_y = torch.cat([fn(inputs_x, inputs_y) for fn in self.embed_fns_y], -1)
        embedding = torch.cat([fn_x, fn_y], -1)

        return embedding

