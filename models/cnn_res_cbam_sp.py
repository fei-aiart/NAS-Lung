import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_sphere import *
from .layers import *

debug = False


class ConvRes(nn.Module):
    """
    model with CBAM, sphere, res
    """
    def __init__(self, config):
        super(ConvRes, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=32, out_channels=64, kernel_size=3)
        self.config = config
        self.last_channel = 64
        self.first_cbam = ResCBAMLayer(64, 32)
        layers = []
        i = 0
        for stage in config:
            i = i + 1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
            layers.append(ResCBAMLayer(self.last_channel, 32 // (2 ** i)))
        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=4, stride=4)
        self.fc = AngleLinear(in_features=self.last_channel, out_features=2)

    def forward(self, inputs):
        if debug:
            print(inputs.size())
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        out = self.first_cbam(out)
        out = self.layers(out)
        if debug:
            print(out.size())
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        if debug:
            print(out.size())
        out = self.fc(out)
        # if debug:
        #     print(out.size())
        return out


def test():
    global debug
    debug = True
    net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    inputs = torch.randn((1, 1, 32, 32, 32))
    output = net(inputs)
    print(net.config)
    print(output)
