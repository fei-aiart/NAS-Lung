import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.last_planes = last_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        self.conv3 = nn.Conv3d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes + dense_depth)
            )

    def forward(self, x):
        # print 'bottleneck_0', x.size(), self.last_planes, self.in_planes, 1
        out = F.relu(self.bn1(self.conv1(x)))
        # print 'bottleneck_1', out.size(), self.in_planes, self.in_planes, 3
        out = F.relu(self.bn2(self.conv2(out)))
        # print 'bottleneck_2', out.size(), self.in_planes, self.out_planes+self.dense_depth, 1
        out = self.bn3(self.conv3(out))
        # print 'bottleneck_3', out.size()
        x = self.shortcut(x)
        d = self.out_planes
        # print 'bottleneck_4', x.size(), self.last_planes, self.out_planes+self.dense_depth, d
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        # print 'bottleneck_5', out.size()
        out = F.relu(out)
        return out

