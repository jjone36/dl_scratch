from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):

    def __init__(self, c_i, c_o, f, s = 1, p = 0):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels = c_i, out_channels = c_o, kernel_size = f,
                              stride = s, padding = p, bais = false)
        self.bn = nn.BatchNorm2d(num_features = c_o, eps = 0.001)
        self.relu = nn.RELU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
