import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_sphere import *


class ResCBAMLayer(nn.Module):
    """
    CBAM+Res model
    """

    def __init__(self, in_planes, feature_size):
        super(ResCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.ch_AvgPool = nn.AvgPool3d(feature_size, feature_size)
        self.ch_MaxPool = nn.MaxPool3d(feature_size, feature_size)
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Softmax = nn.Softmax(1)
        self.sp_Conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sp_Softmax = nn.Softmax(1)

    def forward(self, x):
        x_ch_avg_pool = self.ch_AvgPool(x).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x).view(x.size(0), -1)
        # x_ch_avg_linear = self.ch_Linear2(self.ch_Linear1(x_ch_avg_pool))
        a = self.ch_Linear1(x_ch_avg_pool)
        x_ch_avg_linear = self.ch_Linear2(a)

        x_ch_max_linear = self.ch_Linear2(self.ch_Linear1(x_ch_max_pool))
        ch_out = (self.ch_Softmax(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)) * x
        x_sp_max_pool = torch.max(ch_out, 1, keepdim=True)[0]
        x_sp_avg_pool = torch.sum(ch_out, 1, keepdim=True) / self.in_planes
        sp_conv1 = torch.cat([x_sp_max_pool, x_sp_avg_pool], dim=1)
        sp_out = self.sp_Conv(sp_conv1)
        sp_out = self.sp_Softmax(sp_out.view(x.size(0), -1)).view(x.size(0), 1, x.size(2), x.size(3), x.size(4))
        out = sp_out * x + x
        return out


def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in channels
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: conv3d module
    """
    module = nn.Sequential(

        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=groups,
                  bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module


def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, groups=1,
                     bias=True):
    """
    keep the w,h of inputs same as the outputs

    :param in_channels: num of in in channels
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param dilation: Spacing between kernel elements
    :param groups: Number of blocked connections from input channels to output channels.
    :param bias: If True, adds a learnable bias to the output
    :return: conv3d
    """
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


def conv3d_pooling(in_channels, kernel_size, stride=1,
                   dilation=1, groups=1,
                   bias=False):
    """
    pooling with convolution

    :param in_channels:
    :param kernel_size:
    :param stride:
    :param dilation:
    :param groups:
    :param bias:
    :return: pooling-convolution
    """

    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


class ResidualBlock(nn.Module):
    """
    a simple residual block
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.my_conv1 = make_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.my_conv2 = make_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = make_conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        out1 = self.conv3(inputs)
        out = self.my_conv1(inputs)
        out = self.my_conv2(out)
        out = out + out1
        return out
