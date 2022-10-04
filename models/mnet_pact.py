"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.init as init
from pact_module import ActFn
from torch.autograd import Variable

__all__ = ['MNet_pact']

class MNet(nn.Module):
    def __init__(self, quant, num_classes = 10):
        super().__init__()
        # image size mnist needed for this is 20
        self.conv1 = nn.Conv2d(1, 16, 3) # (20-3)/1+1 = 18
        self.alpha1 = nn.Parameter(torch.tensor(10.))
    
        # maxpool 2,2 = 8  -> (18-2)/2+1 = 9
        self.conv2 = nn.Conv2d(16, 32, 3)  # (9 - 3)/1 + 1 =7  
        self.alpha2 = nn.Parameter(torch.tensor(10.))
        self.ActFn = ActFn.apply

        # maxpool2,2 = 2 -> (7-2)/2+1 = 3
        self.fc1 = nn.Linear(32 * 3 * 3, 10)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(84, num_classes)

        self.pool = nn.MaxPool2d(2,2)
        self.quant = quant()

    def forward(self, x):
        
        x = self.quant(x)
        out = self.ActFn(self.conv1(x), self.alpha1)
        out = self.pool(out)

        # print("after conv1 and pool layer:", out.shape)
        out = self.quant(out)
        out = self.ActFn(self.conv2(out), self.alpha2)
        out = self.pool(out)

        # print("after conv2 and pool layer:", out.shape)
        out = out.view(out.size(0), -1)
        # print("after out.view layer:", out.shape)
        out = self.quant(out)
        out = self.fc1(out)

        # out = self.quant(out)
        # out = self.relu4(self.fc2(out))

        # out = self.quant(out)
        # out = self.fc3(out)

        return out



class MnistBase:
    base = MNet
    args = list()
    kwargs = dict()

class MNet_pact(MnistBase):
    pass


# net = Net()