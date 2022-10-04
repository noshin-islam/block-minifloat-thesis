"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

from quant.quant_module import Quantizer

__all__ = ['VGG16LP', 'VGG16BNLP', 'VGG19LP', 'VGG19BNLP']


def make_layers(cfg, quant, batch_norm=False, conv=nn.Conv2d):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = conv(in_channels, filters, bias=False, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            if use_quant: layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


cfg = {
    16: ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, quant, num_classes=10, depth=16, batch_norm=False):

        self.linear = nn.Linear
        self.conv = nn.Conv2d
        # quant = quant()

        super(VGG, self).__init__()
        # self.quant = quant()
        self.features = make_layers(cfg[depth], quant, batch_norm, self.conv)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.linear(512, 512, bias=False),
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            self.linear(512, 512, bias=False),
            nn.ReLU(True),
            quant(),
            self.linear(512, num_classes, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()

    def forward(self, x):
        #x = x*(1./255.0)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
    def modify_layer_quant(self, new_quant):
        
        for layer in self.features:
            if isinstance(layer, Quantizer):
                # print("changing quant inside self.features")
                layer = new_quant()
        
        for layer in self.classifier:
            if isinstance(layer, Quantizer):
                # print("changing quant inside self.classifier")
                layer = new_quant()
            


class VGGBase:
    base = VGG
    args = list()
    kwargs = dict()

class VGG16LP(VGGBase):
    pass

class VGG16BNLP(VGGBase):
    kwargs = {'batch_norm': True}


class VGG19LP(VGGBase):
    kwargs = {'depth': 19}


class VGG19BNLP(VGGBase):
    kwargs = {'depth': 19, 'batch_norm': True}
