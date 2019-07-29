from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from Blocks import EmptyLayer, DetectionLayer


def parse_cfg(filepath):
    """
    Creating blocks from a cfg file
    input: the file path of .cfg
    output: a list of blocks of the model
    """
    cfg_file = open(filepath, 'r')
    lines = cfg_file.read().split('\n')
    lines = [l for l in lines if len(l) > 0]
    lines = [l for l in lines if l[0] != '#']
    lines = [l.rstrip().lstrip() for l in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, val = line.split('=')
            block[key.rstrip()] = val.lstrip()
    blocks.append(block)                            # for the last one

    return blocks


def create_modules(blocks):
    """
    By taking out the blocks information one by one, create modules accordingly.
    Input: a list of dictionaries of the block attributes
    Output: a module list populated by the modules for convolutional, upsampling, route and yolo layers
    """
    net_setting = blocks[0]
    module_list = nn.ModuleList()
    input_filter = 3
    output_filters = []

    for idx, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # ConvBlock
        if block['type'] == 'convolutional':
            activation = block['activation']

            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            padding = int(block['pad'])

            if padding:
                pad = (kernel_size - 1) // 2  # SAME
            else:
                pad = 0                       # VALID

            # ConvLayer
            conv_layer = nn.Conv2d(in_channels = input_filter, out_channels = filters,
                                    kernel_size = kernel_size, stride = stride, padding = pad, bias = bias)
            module.add_module('conv_{}'.format(idx), conv_layer)

            # BNLayer
            if batch_normalize:
                bn_layer = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(idx), bn_layer)

            # Activation layer
            if activation:
                act_layer = nn.LeakyReLU(.1, inplace = True)
                module.add_module('leaky_{}'.format(idx), act_layer)

        # Upsampling
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            module.add_module('upsample_{}'.format(idx), upsample)

        # Skip connection
        elif block['type'] == 'shortcut':
            from_layer = int(block['from'])
            skip_layer = EmptyLayer()
            module.add_module('shortcut_{}'.format(idx), skip_layer)

        # Route Layer
        elif block['type'] == 'route':
            layer_nums = block['layers'].split(',')
            start = int(layer_nums[0])

            try:
                end = int(layer_nums[1])
            except:
                end = 0

            route_layer = EmptyLayer()
            module.add_module("route_{}".format(idx), route_layer)

            # Copying the filters from the its previous layer
            if end == 0:
                filters = output_filters[start]
            else:
                filters = output_filters[start] + output_filters[end]

        # Detection Layer
        elif block['type'] == 'yolo':
            mask = [int(m) for m in block['mask'].split(',')]

            anchors = [int(a) for a in block['anchors'].split(',')]
            anchors = [(anchors[2*i], anchors[2*i+1]) for i in range(9)]
            anchors = [anchors[i] for i in mask]

            detect_layer = DetectionLayer(anchors)
            module.add_module('detection_{}'.format(idx), detect_layer)

        module_list.append(module)
        input_filter = filters
        output_filters.append(filters)

    return (net_setting, module_list)



if __name__=='__main__':

    blocks = parse_cfg('yolo_v3.cfg')
    net_setting, module_list = create_modules(blocks)

    # Create Darknet class
    model = Darknet(blocks, net_setting, module_list)

    # Download Here: https://pjreddie.com/media/files/yolov3.weights
    model.load_weights('../data/yolov3.weights')

    # Load the test image
    input = load_test_data('ex.jpg')
    pred = model.forward(input, torch.cuda.is_available())
    print(pred)
