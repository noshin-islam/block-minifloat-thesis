"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['MNet', 'MNetSmall']

class MNet(nn.Module):
    def __init__(self, quant, num_classes = 10):
        super().__init__()
        # image size mnist needed for this is 20
        self.conv1 = nn.Conv2d(1, 16, 3) # (20-3)/1+1 = 18
        self.relu1 = nn.ReLU()
        # maxpool 2,2 = 8  -> (18-2)/2+1 = 9
        self.conv2 = nn.Conv2d(16, 32, 3)  # (9 - 3)/1 + 1 =7  
        self.relu2 = nn.ReLU()
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

        out = self.pool(self.relu1(self.conv1(x)))
        # print("after conv1 and pool layer:", out.shape)
        out = self.quant(out)
        out = self.pool(self.relu2(self.conv2(out)))
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

class MNetSmall(nn.Module):
    def __init__(self, quant, num_classes = 10):
        super().__init__()

        # self.conv1 = nn.Conv2d(1, 2, 1) #nn.Conv2d(3, 6, 5) edit -> (20 - 5)/1 + 1 = 16
        # # 4x4 -> (4-1)/1 + 1 = 4
        # # (15-3)/1+1 = 13
        # self.relu1 = nn.ReLU()
        
        # self.fc1 = nn.Linear(2 * 4 * 4, num_classes)
    
        # self.quant = quant()

        self.fc1 = nn.Linear(1 * 5 * 5, 60)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(60, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, num_classes)
        self.quant = quant()

    def forward(self, x):

        #this is for the single layer conv net from before

        # print("INPUT SHAPE: ", x.shape)
        # out = self.quant(x)
        # out = self.conv1(out)
        # out = self.relu1(out)
        # # print("HERE: ",out.shape)
        # out = out.view(out.size(0), -1)
        # # print("HERE1: ", out.shape)
        # out = self.quant(out)
        # out = self.fc1(out)

        # 3 layer perceptron

        out = x.view(x.size(0), -1)

        out = self.quant(out)
        out = self.relu1(self.fc1(out))

        out = self.quant(out)
        out = self.relu2(self.fc2(out))

        out = self.quant(out)
        out = self.fc3(out)

        return out

class MnistBase:
    base = MNet
    args = list()
    kwargs = dict()

class MNet(MnistBase):
    pass

class MNetSmall(MnistBase):
    base = MNetSmall

# net = Net()