# Origial Paper : https://arxiv.org/pdf/1610.02357.pdf

import numpy as np
import tensorflow as tf


def init_filter(f, mi, mo, stride):
    '''
    initialize filters
    if input_shape = (a, a, mi), fm_sizes = (f, f, mi), stride = s,
    output_shape = ( (a-f)/s +1, (a-f)/s +1, mo ) where mo = number of filters
    '''
    num = np.random.randn(f, f, mi, mo) * np.sqrt(2./(f*f*mi))
    return num.astype(np.float32)

# Convolutional layer class
class ConvLayer:

    def __init__(self, f, mi, mo, stride = 2, padding = 'VALID'):
        self.W = tf.Variable(init_filter(f, mi, mo, stride))
        self.b = tf.Variable(np.zeros(mo, dtype = np.float32))
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        X = tf.nn.conv2d(input = X,
                        filter = self.W,
                        strides = [1, self.stride, self.stride, 1],
                        padding = self.padding)
        X += self.b
        return X

    # This is for sanity check later
    def copy_keras_layers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))

    def get_params(self):
        return self.W, self.b


# BatchNormalization layer class
class BNLayer:

    def __init__(self, D):
        self.running_mean = tf.Variable(np.zeros(D, dtype = np.float32), trainable = True)
        self.running_var = tf.Variable(np.zeros(D, dtype = np.float32), trainable = True)
        self.beta = tf.Variable(np.zeors(D, dtype = np.float32))
        self.gamma = tf.Variable(np.zeros(D, dtype = np.float32))

    def forward(self, X):
        return tf.nn.batch_normalization(X,
                                        mean = self.running_mean,
                                        variance = self.running_var,
                                        offset = self.beta,
                                        scale = self.gamma,
                                        variance_epsilon = 1e-3)

    # This is for sanity check later
    def copy_keras_layers(self, layer):
        gamma, beta, running_mean, running_var = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_var.assign(running_var)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1, op2, op3, op4))

    def get_params(self):
        return [self.running_mean, self.running_var, self.beta, self.gamma]


class SeparableConv:

    def __init__(self, mi, ):
