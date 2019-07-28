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
