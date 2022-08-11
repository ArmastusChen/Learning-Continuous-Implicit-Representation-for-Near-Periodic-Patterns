import torch
import torch.nn as nn
import torchvision.models as models
from copy import deepcopy


'''the various activation functions defined as subclasses of nn.Module'''
class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class SinPlusCosActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x) + torch.cos(x)

class XPlusSinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.sin(x)

class SnakeActivation(nn.Module):
    def __init__(self, a=1):
        super().__init__()
        self.a = a

    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x))/self.a

class LearnedSnake(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.a.requiresGrad = True

    def forward(self, x):
        return x + torch.square(torch.sin(self.a * x))/self.a


'''a basic feedforward neural network returning a model with the specified architecture and activations
e.g. SimpleMLP((1, 64, 64, 1), "snake")'''

class SimpleMLP(nn.Module):
    def __init__(self, layer_sizes, hidden_activation):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        for k in range(len(layer_sizes)-2):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
            if hidden_activation == "tanh":
                self.layers.append(nn.Tanh())
            elif hidden_activation == "relu":
                self.layers.append(nn.ReLU())
            elif hidden_activation == "sin":
                self.layers.append(SinActivation())
            elif hidden_activation == "sin_plus_cos":
                self.layers.append(SinPlusCosActivation())
            elif hidden_activation[:5] == "snake":
                if hidden_activation == "snake":
                    a = 1
                else:
                    a = float(hidden_activation[8:])
                self.layers.append(SnakeActivation(a))
            elif hidden_activation == "learned_snake":
                self.layers.append(LearnedSnake())
            elif hidden_activation == "x_sin":
                self.layers.append(XPlusSinActivation())
            else:
                raise Exception("Unknown activation")
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

'''ResNet18 with the specified activation. Since the CIFAR images
 aren't very large, the initial conv1 and maxpool layers are modified.'''

def ResNet18WithActivation(activation):
    resnet18 = models.resnet18()
    resnet18.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet18.maxpool = torch.nn.Identity()

    resnet18.relu = deepcopy(activation)
    resnet18.layer1[0].relu = deepcopy(activation)
    resnet18.layer1[1].relu = deepcopy(activation)
    resnet18.layer2[0].relu = deepcopy(activation)
    resnet18.layer2[1].relu = deepcopy(activation)
    resnet18.layer3[0].relu = deepcopy(activation)
    resnet18.layer3[1].relu = deepcopy(activation)
    resnet18.layer4[0].relu = deepcopy(activation)
    resnet18.layer4[1].relu = deepcopy(activation)

    resnet18.fc = nn.Linear(512, 10)

    return resnet18

'''a simple RNN used for comparison with snake MLPs'''
class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1, activation="relu"):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.rnn = nn.RNN(input_size, hidden_layer_size, nonlinearity=activation)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size)

    def forward(self, input_seq):
        if torch.cuda.is_available():
            self.hidden_cell = self.hidden_cell.cuda()
        rnn_out, self.hidden_cell = self.rnn(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)

        predictions = self.linear(rnn_out.view(len(input_seq), -1))
        return predictions[-1]
