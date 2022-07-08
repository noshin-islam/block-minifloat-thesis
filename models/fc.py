"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['SLayer', 'SingleLayer']

class SingleLayer(nn.Module):
    def __init__(self, quant, num_classes = 1):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu1 = nn.ReLU()
        self.quant = quant()

    def forward(self, x):

        out = x.view(x.size(0), -1)

        out = self.quant(out)
        out = self.relu1(self.fc1(out))

        return out

class SLayer:
    base = SingleLayer
    args = list()
    kwargs = dict()
class SingleLayer(SLayer):
    base = SingleLayer

