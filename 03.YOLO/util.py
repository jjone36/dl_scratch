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


def predict_transform(pred, im_size, anchors, n_classes, CUDA = True):
    """
    Converting the feature map from the detection layer to 2D tensor
    Input  -pred: the output feature map from the previous convolutional layer
           -im_size: the dimension of input images (416*416)
           -anchors: yolo priors
           -n_classes: number of classes of the dataset (80)
    Output -2D tensor of the predicion, (4 bbox coordinates + class prediction) for each of 3 boxes
    """
    batch_size = pred.size(0)
    stride =  im_size // pred.size(2)
    grid_size = im_size // stride
    bbox_attrs = 5 + n_classes
    num_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    pred = pred.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    pred = pred.transpose(1,2).contiguous()
    pred = pred.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    pred[:,:,0] = torch.sigmoid(pred[:,:,0])
    pred[:,:,1] = torch.sigmoid(pred[:,:,1])
    pred[:,:,4] = torch.sigmoid(pred[:,:,4])

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    pred[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4])*anchors

    #Softmax the class scores
    pred[:,:,5: 5 + n_classes] = torch.sigmoid((pred[:,:, 5 : 5 + n_classes]))
    pred[:,:,:4] *= stride

    return pred


def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]


    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))


        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)

        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask

                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    return output
