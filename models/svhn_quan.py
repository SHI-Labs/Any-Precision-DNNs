# Adapted from
# https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/nets/cifar_resnet.py
# https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/svhn-digit-dorefa.py

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn

__all__ = ['svhnq']


class Activate(nn.Module):
    def __init__(self, bit_list, quantize=True):
        super(Activate, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(bit_list=self.bit_list)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x


class SVHNQ(nn.Module):
    def __init__(self, bit_list, num_classes, expand=8):
        super(SVHNQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        Conv2d = conv2d_quantize_fn(bit_list=self.bit_list)
        NormLayer = batchnorm_fn(self.bit_list)
        self.expand = expand

        ep = self.expand
        self.layers = nn.Sequential(
            nn.Conv2d(3, ep * 6, 5, padding=0, bias=True),
            nn.MaxPool2d(2),
            Activate(self.bit_list),
            # 18
            Conv2d(ep * 6, ep * 8, 3, padding=1, bias=False),
            NormLayer(ep * 8),
            Activate(self.bit_list),
            Conv2d(ep * 8, ep * 8, 3, padding=1, bias=False),
            NormLayer(ep * 8),
            nn.MaxPool2d(2),
            Activate(self.bit_list),
            # 9
            Conv2d(ep * 8, ep * 16, 3, padding=0, bias=False),
            NormLayer(ep * 16),
            Activate(self.bit_list),
            # 7
            Conv2d(ep * 16, ep * 16, 3, padding=1, bias=False),
            NormLayer(ep * 16),
            Activate(self.bit_list),
            Conv2d(ep * 16, ep * 16, 3, padding=0, bias=False),
            NormLayer(ep * 16),
            Activate(self.bit_list),
            # 5
            nn.Dropout(0.5),
            Conv2d(ep * 16, ep * 64, 5, padding=0, bias=False),
            Activate(self.bit_list, quantize=False),
        )
        self.fc = nn.Linear(ep * 64, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view([x.shape[0], -1])
        x = self.fc(x)
        return x


def svhnq(bit_list, num_classes=10, expand=2):
    return SVHNQ(bit_list, num_classes=num_classes, expand=expand)
