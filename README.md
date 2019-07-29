# Deep Dive into the Computer Vision World
*: repo for implementing the neural networks from scratch*

<br>

## ***1. Paper Reproduction***

Studying neural networks can be divided into three parts: *The applications*, *the implementations*, and *intuitions behind these architectures*. Thanks to the user-friendly frameworks such Keras, the applications part are open to everyone. But grasping the real intuition behind the model is overlooked sometimes. What’s the researchers’ intention for building a model with such structures? What motivated them to take such an approach? And what can we infer from the outcome?  

![page](https://github.com/jjone36/dl_scratch/blob/master/data/img.png)

This repository is an on-going project for studying the state-of-the-art networks. Starting from VGG, the intuitions and implementation of networks will be covered. The networks are mostly focused on the milestones in Computer Vision such as Image Classification, Object Detections, Image Segmentation, Face Detections etc.

<br>

* **Project Date:** Jul 2019 ~
* **Applied skills:** Tensorflow, Keras, PyTorch

<br>

## ***2. Paper Reviews***

- Part1. [VGG, ResNet, Inception Network, Xception and MobileNet](https://towardsdatascience.com/deep-dive-into-the-computer-vision-world-f35cd7349e16?source=friends_link&sk=449ea5da20c884dadca23b907efb7e13)
- Part2. [R-CNN, Fast R-CNN, and Faster R-CNN](https://towardsdatascience.com/deep-dive-into-the-computer-vision-world-part-2-7a24efdb1a14?source=friends_link&sk=4fec4dfc9499c930f263c6808b2f369d)
- Part3. [YOLO and SSD, Mask R-CNN]()

<br>

## ***3. Implementation From Scratch***

- [VGG in Tensorflow](https://github.com/jjone36/dl_scratch/blob/master/vgg_tf.py)
- [ResNet in Tensorflow](https://github.com/jjone36/dl_scratch/blob/master/01.ResNet/ResNet.py) and [Keras applications](https://github.com/jjone36/dl_scratch/blob/master/01.ResNet/resnet_transfer.py)
- [Inception-V1 in Keras](https://github.com/jjone36/dl_scratch/blob/master/02.InceptionNetwork/inception_v1_keras.py)
- [Inception-V4 in Pytorch](https://github.com/jjone36/dl_scratch/blob/master/02.InceptionNetwork/inception_v4_pytorch.py)
- [Inception-ResNet V2 in Tensorflow](https://github.com/jjone36/dl_scratch/blob/master/02.InceptionNetwork/inception-resnet_v2_tf.py)
- [YOLO V3 in PyTorch](https://github.com/jjone36/dl_scratch/tree/master/03.YOLO)

<br>

## ***4. Reference***
- Karen Simonyan and Andrew Zisserman, [Very deep convolutional network for large-scale image recognition](https://arxiv.org/abs/1409.1556), 2015
- Kaiming He, et al. [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385), 2015
- Christian Szegedy, et al. [Going deeper with convolutions](https://arxiv.org/abs/1409.4842), 2014
- Christian Szegedy et al., [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf), 2016
- Franc¸ois Chollet, [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf), 2017
- Andrew G. Howard et al., [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf), 2017
- Ross Girshick et al, [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf), 2014
- Kaiming He et al, [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf), 2015
- Ross Girshick, [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf), 2015
- Shaoqing Ren et al, [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf), 2015
- Joseph Redmon et al. [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf), 2016
- Joseph Redmon et al. [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf), 2016
- Joseph Redmon et al. [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), 2018
