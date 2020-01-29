# Refer to https://arxiv.org/abs/1512.03385
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn

__all__ = ['resnet20q', 'resnet50q']


class Activate(nn.Module):
    def __init__(self, bit_list, quantize=True):
        super(Activate, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.bit_list)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x


class PreActBasicBlockQ(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    def __init__(self, bit_list, in_planes, out_planes, stride=1):
        super(PreActBasicBlockQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.bit_list)
        NormLayer = batchnorm_fn(self.bit_list)

        self.bn0 = NormLayer(in_planes)
        self.act0 = Activate(self.bit_list)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(out_planes)
        self.act1 = Activate(self.bit_list)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.bn0(x)
        out = self.act0(out)

        if self.skip_conv is not None:
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x

        out = self.conv0(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv1(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_units, bit_list, num_classes, expand=5):
        super(PreActResNet, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.expand = expand

        NormLayer = batchnorm_fn(self.bit_list)

        ep = self.expand
        self.conv0 = nn.Conv2d(3, 16 * ep, kernel_size=3, stride=1, padding=1, bias=False)

        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep
        self.layers = nn.ModuleList()
        for stride, channel in zip(strides, channels):
            self.layers.append(block(self.bit_list, in_planes, channel, stride))
            in_planes = channel

        self.bn = NormLayer(64 * ep)
        self.fc = nn.Linear(64 * ep, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.fc(out)
        return out


class PreActBottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, bit_list, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBottleneckQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.bit_list)
        norm_layer = batchnorm_fn(self.bit_list)

        self.bn0 = norm_layer(in_planes)
        self.act0 = Activate(self.bit_list)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_planes)
        self.act1 = Activate(self.bit_list)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.act2 = Activate(self.bit_list)
        self.conv2 = Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample

    def forward(self, x):        
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = self.conv0(self.act0(self.bn0(x)))
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out


class PreActResNetBottleneck(nn.Module):
    def __init__(self, block, layers, bit_list, num_classes):
        super(PreActResNetBottleneck, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        self.norm_layer = batchnorm_fn(self.bit_list)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn = self.norm_layer(512 * block.expansion)
        self.act = Activate(self.bit_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.bit_list, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.bit_list, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.act(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# For CIFAR10
def resnet20q(bit_list, num_classes=10):
    return PreActResNet(PreActBasicBlockQ, [3, 3, 3], bit_list, num_classes=num_classes)


# For ImageNet
def resnet50q(bit_list, num_classes=1000):
    return PreActResNetBottleneck(PreActBottleneckQ, [3, 4, 6, 3], bit_list, num_classes=num_classes)
