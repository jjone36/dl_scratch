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


    def save_weights():
        pass


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
