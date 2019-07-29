from __future__ import division
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



class Darknet(nn.Module):

    def __init__(self, blocks, net_setting, module_list):
        super(Darknet, self).__init__()
        self.blocks = blocks
        self.net_setting = net_setting
        self.module_list = module_list

    def forward(self, x, CUDA):
        blocks = self.blocks[1:]
        outputs = {}                # for caching the layers for EmptyLayers
        detections = []
        temp = 0

        for idx, block in enumerate(blocks):

            type = block['type']
            if (type == 'convolutional') or (type == 'upsample'):
                x = self.module_list[idx](x)
                outputs[idx] = x

            elif type == 'shortcut':
                n = int(block['from'])
                x = outputs[idx-1] + outputs[idx+n]
                outputs[idx] = x

            elif type == 'route':
                layers = [int(i) for i in block['layers'].split(',')]
                if len(layers) == 1:
                    x = outputs[idx + layers[0]]
                else:
                    x_1 = outputs[idx + layers[0]]
                    x_2 = outputs[layers[1]]
                    x = torch.cat((x_1, x_2), 1)
                outputs[idx] = x

            elif type == 'yolo':
                # get the attibutes
                anchors = self.module_list[idx][0].anchors
                im_size = int(self.net_setting['height'])
                n_classes = int(block['classes'])
                outputs[idx] = x

                # Convert the feature map into 2D tensor
                x = x.data
                x = predict_transform(x, im_size, anchors, n_classes, CUDA)

                if temp == 0:
                    detections = x
                    temp = 1
                else:
                    detections = torch.cat((detections, x), 1)

        return detections

    def load_weights(weight_file):

        fp = open(weight_file, 'rb')
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        weights = np.fromfile(fp, dtype = np.float32)

        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        blocks = self.blocks[1:]
        pos = 0
        for idx, block in enumerate(blocks):

            if block['type'] == 'convolutional':
                layers = self.module_list[idx]

                # Load the weights for BN Layer
                bn_layer = layer[1]
                a = len(bn_layer.bias)

                bn_bias = torch.from_numpy(weights[pos:pos+a])
                pos += a

                bn_weight = torch.from_numpy(weights[pos:pos+a])
                pos += a

                bn_mean = torch.from_numpy(weights[pos:pos+a])
                pos += a

                bn_var = torch.from_numpy(weights[pos:pos+a])
                pos += a

                # Reshaping the dims of the values
                bn_bias = bn_bias.view(bn_layer.bias.data.size())
                bn_weight = bn_weight.view(bn_layer.weight.data.size())
                bn_mean = bn_mean.view(bn_layer.running_mean.data.size())
                bn_var = bn_var.view(bn_layer.running_var.data.size())

                # Copy the values to the model
                bn_layer.bias.copy_(bn_bias)
                bn_layer.weight.copy_(bn_weight)
                bn_layer.running_mean.copy_(bn_mean)
                bn_layer.running_var.copy_(bn_var)

                # Load the weights for Conv Layer
                conv_layer = layer[0]
                a = len(conv_layer.weight)
                conv_weight = torch.from_numpy(weights[pos:pos+a])
                conv_weight = conv_weight.view(conv_layer.weight.data.size())
                conv_layer.weight.copy_(conv_weight)
                pos += a
