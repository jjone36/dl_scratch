# Reference: https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/util.py

from __future__ import division

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def load_test_data(img_file):
    """
    Loading and processing an input image for testing the model
    Input: img_file
    Output: the processed image
    """
    img = cv2.imread(img_file)
    img = cv2.resize(img, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))       # Channel - hight - width
    img = img[None, :, :, :]/255
    img = Variable(torch.from_numpy(img).float(), requires_grad = True)
    return input


def predict_transform(prediction, im_size, anchors, n_classes, CUDA = True):
    """
    Converting the feature map from the detection layer to 2D tensor
    Input -pred: the output feature map from the previous convolutional layer
    -im_size: the dimension of input images (416*416)
    -anchors: yolo priors
    -n_classes: number of classes of the dataset (80)
    Output -2D tensor of the predicion, (4 bbox coordinates + class prediction) for each of 3 boxes
    """
    batch_size = prediction.size(0)
    stride =  im_size // prediction.size(2)
    grid_size = im_size // stride
    bbox_attrs = 5 + n_classes
    num_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores
    prediction[:,:,5: 5 + n_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + n_classes]))

    prediction[:,:,:4] *= stride

    return prediction


def drawing_box():
    pass
