"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['SLayer', 'SingleLayer']

class SingleLayer(nn.Module):
    def __init__(self, quant, num_classes = 2):
        super().__init__()
        self.fc1 = nn.Linear(6, 2)
        self.relu1 = nn.ReLU()
        self.quant = quant()

    def forward(self, x):
        print('x_shape:',x.shape)
        out = x.view(x.size(0), -1)
        print('out_shape:',out.shape)
        out = self.quant(out)
        out = self.relu1(self.fc1(out))

        return out

class SLayer:
    base = SingleLayer
    args = list()
    kwargs = dict()
class SingleLayer(SLayer):
    base = SingleLayer

