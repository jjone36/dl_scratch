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


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch_0 = nn.Sequential(
                            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1, count_include_pad = False)
                            Conv_block(c_i = 384, c_o= 96, f=1)
                            )
        self.branch_1 = Conv_block(c_i = 384, c_o= 96, f= 1)
        self.branch_2 = nn.Sequential(
                            Conv_block(c_i = 384, c_0 = 64, f = 1)
                            Conv_block(c_i = 64, c_o = 96, f = 3, p = 1)
                            )
        self.branch_3 = nn.Sequential(
                            Conv_block(c_i = 384, c_0 = 64, f = 1)
                            Conv_block(c_i = 64, c_o = 96, f = 3, p = 1)
                            Conv_block(c_i = 96, c_o = 96, f = 3, p = 1)
                            )

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        return torch.cat((x_0, x_1, x_2, x_3), 1)


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch_0 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.branch_1 = Conv_block(c_i = 384, c_o= 384, f = 3, stride = 2)
        self.branch_2 = nn.Sequential(
                            Conv_block(c_i= 384, c_o= 192, f = 1)
                            Conv_block(c_i = 192, c_o= 224, f = 3, p = 1)
                            Conv_block(c_i= 224, c_o= 256, f = 3, s =2)
                            )

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        return torch.cat((x_0, x_1, x_2), 1)


class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch_0 = nn.Sequential(
                            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False)
                            Conv_block(c_i=1024, c_o=128, f = 1)
                            )
        self.branch_1 = Conv_block(c_i=1024, c_o=384, f=1)
        self.branch_2 = nn.Sequential(
                            Conv_block(c_i=1024, c_o=192, f=1)
                            Conv_block(c_i=192, c_o=224, f=(1, 7), p = (0, 3))
                            Conv_block(c_i=224, c_o=256, f=(7, 1), p = (3, 0))
                            )
        self.branch_3 = nn.Sequential(
                            Conv_block(c_i=1024, c_o=192, f=1)
                            Conv_block(c_i=192, c_o=192, f=(1, 7), p = (0, 3))
                            Conv_block(c_i=192, c_o=224, f=(7, 1), p = (3, 0))
                            Conv_block(c_i=224, c_o=224, f=(1, 7) p = (0, 3))
                            Conv_block(c_i=224, c_o=256, f=(7, 1), p = (3, 0))
                            )

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        return torch.cat((x_0, x_1, x_2, x_3), 1)


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.branch_1 = nn.Sequential(
                            Conv_block(c_i = 1024, c_o=192, f = 1)
                            Conv_block(c_i = 192, c_o=192, f = 3, s = 2)
                            )
        self.branch_2 = nn.Sequential(
                            Conv_block(c_i = 1024, c_o=256, f = 1)
                            Conv_block(c_i = 256, c_o=256, f = (1, 7), p = (0, 3))
                            Conv_block(c_i = 256, c_o=320, f = (7, 1), p = (3, 0))
                            Conv_block(c_i = 320, c_o=320, f = 3, s = 2)
                            )

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        return torch.cat((x_0, x_1, x_2), 1)


class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__():
        self.branch_0 = nn.Sequential(
                            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False)
                            Conv_block(c_i= 1536, c_o=384, f=1)
                            )
        self.branch_1 = Conv_block(c_i= 1536, c_o=256, f=1)
        self.branch_2 = Conv_block(c_i= 1536, c_o=384, f=1)
        self.branch_2_a = Conv_block(c_i= 384, c_o=256, f= (1, 3), p = (1, 0))
        self.branch_2_b = Conv_block(c_i= 384, c_o=256, f= (3, 1), p = (0, 1))
        self.branch_3 = nn.Sequential(
                            Conv_block(c_i= 1536, c_o=384, f=1)
                            Conv_block(c_i= 384, c_o=448, f=(1, 3), p = (0, 1))
                            Conv_block(c_i= 448, c_o=512, f=(3, 1), p = (1, 0))
                            )
        self.branch_3_a = Conv_block(c_i= 512, c_o= 256, f=(3, 1), p = (1, 0))
        self.branch_3_b = Conv_block(c_i= 512, c_o= 256, f=(1, 3), p = (0, 1))

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)

        x_2 = self.branch_2(x)
        x_2_a = self.branch_2_a(x_2)
        x_2_b = self.branch_2_b(x_2)
        x_2 = torch.cat((x_2_a, x_2_b), 1)

        x_3 = self.branch_3(x)
        x_3_a = self.branch_3_a(x_3)
        x_3_b = self.branch_3_b(x_3)
        x_3 = torch.cat((x_3_a, x_3_b), 1)
        return torch.cat((x_0, x_1, x_2, x_3), 1)


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.layer_0 = Conv_block(c_i = 3, c_o = 32, f = 3, s = 2)
        self.layer_1 = Conv_block(c_i = 32, c_o = 32, f = 3)
        self.layer_2 = Conv_block(c_i = 32, c_o = 64, f = 3, p = 1)

        self.layer_3_a = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.layer_3_b = Conv_block(c_i = 64, c_o = 96, f = 3, s = 2)

        self.layer_4_a = nn.Sequential(
                                Conv_block(c_i = 160, c_o = 64, f = 1)
                                Conv_block(c_i = 64, c_o = 96, f = 3)
                                )
        self.layer_4_b = nn.Sequential(
                                Conv_block(c_i = 160, c_o = 64, f = 1, p = 1)
                                Conv_block(c_i = 64, c_o = 64, f = (7, 1), p = (3, 0))
                                Conv_block(c_i = 64, c_o = 64, f = (1, 7), p = (0, 3))
                                Conv_block(c_i = 64, c_o = 96, f = 3)
                                )
        self.layer_5_a = Conv_block(c_i = 192, c_o = 192, f = 3)
        self.layer_5_b = nn.MaxPool2d(kernel_size = 3, stride = 2)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x_a = self.layer_3_a(x)
        x_b = self.layer_3_b(x)
        x = torch.cat((x_a, x_b), 1)

        x_a = self.layer_4_a(x)
        x_b = self.layer_4_b(x)
        x = torch.cat((x_a, x_b), 1)

        x_a = self.layer_5_a(x)
        x_a = self.layer_5_b(x)
        return torch.cat((x_a, x_b), 1)


class Inception_v4(nn.Module):
    def __init__(self, im_size = 299, n_classes = 1001):
        super(Inception_v4, self).__init__()
        self.input_space = None
        self.mean = None
        self.std = None
        self.input_size = (im_size, im_size, 3)
        self.features = nn.Sequential(
                    Stem()
                    # Module A
                    Inception_A()
                    Inception_A()
                    Inception_A()
                    Inception_A()
                    Reduction_A()
                    # Module B
                    Inception_B()
                    Inception_B()
                    Inception_B()
                    Inception_B()
                    Inception_B()
                    Inception_B()
                    Inception_B()
                    Reduction_B()
                    # Module C
                    Inception_C()
                    Inception_C()
                    Inception_C()
                    )
        self.last_layer = nn.Linear(in_features = 1536, out_features = n_classes)

    def forward(self, images):
        x = self.features(images)
        x = F.avg_pool2d(input = x, kernel_size = 8)
        x = F.dropout(x, p = .8, training = True)
        x = x.view(x.size(0), -1)
        return self.last_layer(x)
