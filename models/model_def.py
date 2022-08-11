#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import contextlib

__all__ = [
    'get_available_models', 'get_model_def',
    'ModelDef', 'AlexNetModelDef', 'AlexNet5ModelDef', 'AlexNet5Resize256x256ModelDef', 'ResNet34ModelDef',
    'FeatureExtractor', 'HookBasedFeatureExtractor'
]

import os

import sys


def get_available_models():
    return ['alexnet', 'alexnet5', 'alexnet5resize256x256', 'ResNet34ModelDef']


def get_model_def(name):
    if name == 'alexnet':
        return AlexNetModelDef()
    elif name == 'alexnet5':
        return AlexNet5ModelDef()
    elif name == 'alexnet5resize256x256':
        return AlexNet5Resize256x256ModelDef()
    elif name == 'resnet34':
        return ResNet34ModelDef()
    elif name == 'resnet18':
        return ResNet18ModelDef()
    elif name == 'vgg19':
        return Vgg19ModelDef()
    else:
        raise NotImplementedError('Unknown model name: {}.'.format(name))


class ModelDef(object):
    def __init__(self):
        super().__init__()

    def get_model(self, use_gpu):
        model = self._get_model()
        model.eval()
        if use_gpu:
            model.cuda()
        return model

    def _get_model(self, ckpt_path=None):
        raise NotImplementedError()


class FeatureExtractor(object):
    pass


class HookBasedFeatureExtractor(FeatureExtractor):
    def __init__(self, model, activations):
        super().__init__()
        self.model = model
        self.activations = activations

    def __call__(self, input):
        self.model(input)
        return self.activations.copy()


class DecoratorBasedFeatureExtractor(FeatureExtractor):
    def __init__(self, model, used_names):
        super().__init__()
        self.model = model
        self.used_names = used_names

    def __call__(self, input):
        output = self.model(input)
        output = {k: v for k, v in output.items() if k in self.used_names}
        assert len(output) == len(self.used_names)
        return output


class AlexNetModelDef(ModelDef):
    input_size = 224

    nr_convs = 1
    conv_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5'][:1]
    conv_dims = [64, 384, 256, 256, 256][:1]
    conv_downsamples = [4, 8, 16, 16, 16][:1]
    peak_window_size = [11, 7, 3, 3, 3][:1]
    disp_alpha = [5.0, 7.0, 15.0, 15.0, 15.0][:1]
    inclusions = [i for i in range(conv_dims[0]) if i not in [1, 2, 4, 15, 19, 20, 21, 35, 36, 37, 38, 40, 49, 51, 52, 53, 58]]

    _conv_layer_ids = [0, 3, 6, 8, 10][:1]
    def _get_model(self):
        from .alexnet import alexnet
        print(os.getcwd())
        return alexnet(True, ckpt_path='alexnet-owt-4df8aa71.pth')

    @contextlib.contextmanager
    def hook_model(self, model):
        activations = [None for _ in range(self.nr_convs)]
        handles = [None for _ in range(self.nr_convs)]

        for i in range(self.nr_convs):
            self._set_hook(model, i, activations, handles)

        yield HookBasedFeatureExtractor(model, activations)

        for h in handles:
            h.remove()

    def _set_hook(self, model, i, activations, handles):
        def fetch(self, input, output):
            activations[i] = output.data.clone()

        handles[i] = model.features[self._conv_layer_ids[i]].register_forward_hook(fetch)
#
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(5, 5))
#   (1): ReLU(inplace=True)
#   (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (4): ReLU(inplace=True)
#   (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (7): ReLU(inplace=True)
#   (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (9): ReLU(inplace=True)
#   (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
# )
class AlexNet5ModelDef(AlexNetModelDef):
    input_size = 256
    nr_convs = 5
    conv_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    conv_dims = [64, 192, 384, 256, 256]
    conv_downsamples = [4, 8, 16, 16, 16]
    peak_window_size = [11, 7, 3, 3, 3]
    disp_alpha = [5.0, 7.0, 15.0, 15.0, 15.0]

    _conv_layer_ids = [0, 3, 6, 8, 10]


class AlexNet5Resize256x256ModelDef(AlexNet5ModelDef):
    input_size = (256, 256)


class ResNet34ModelDef(ModelDef):
    nr_convs = 1
    def _get_model(self, ckpt_path=None):
        from .resnet import resnet34
        assert ckpt_path is None
        return resnet34(pretrained=True)

    @contextlib.contextmanager
    def hook_model(self, model):
        activations = [None for _ in range(self.nr_convs)]
        handles = [None for _ in range(self.nr_convs)]

        for i in range(self.nr_convs):
            self._set_hook(model, i, activations, handles)

        yield HookBasedFeatureExtractor(model, activations)

        for h in handles:
            h.remove()

    def _set_hook(self, model, i, activations, handles):
        def fetch(self, input, output):
            activations[i] = output.data.clone()

        handles[i] = model.conv1.register_forward_hook(fetch)



class ResNet18ModelDef(ModelDef):
    nr_convs = 1

    def _get_model(self, ckpt_path=None):
        from .resnet import resnet18
        assert ckpt_path is None
        return resnet18(pretrained=True)

    @contextlib.contextmanager
    def hook_model(self, model):
        activations = [None for _ in range(self.nr_convs)]
        handles = [None for _ in range(self.nr_convs)]

        for i in range(self.nr_convs):
            self._set_hook(model, i, activations, handles)

        yield HookBasedFeatureExtractor(model, activations)

        for h in handles:
            h.remove()

    def _set_hook(self, model, i, activations, handles):
        def fetch(self, input, output):
            activations[i] = output.data.clone()

        handles[i] = model.conv1.register_forward_hook(fetch)




class Vgg19ModelDef(ModelDef):
    nr_convs = 1
    _conv_layer_ids = [0, 3, 6, 8, 10]

    def _get_model(self, ckpt_path=None):
        from .vgg import vgg19
        assert ckpt_path is None
        return vgg19(pretrained=True)

    @contextlib.contextmanager
    def hook_model(self, model):
        activations = [None for _ in range(self.nr_convs)]
        handles = [None for _ in range(self.nr_convs)]

        for i in range(self.nr_convs):
            self._set_hook(model, i, activations, handles)

        yield HookBasedFeatureExtractor(model, activations)

        for h in handles:
            h.remove()

    def _set_hook(self, model, i, activations, handles):
        def fetch(self, input, output):
            activations[i] = output.data.clone()
        handles[i] = model.features[self._conv_layer_ids[i]].register_forward_hook(fetch)