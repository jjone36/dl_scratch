
import numpy as np
import tensorflow as tf
import tf.contrib.slim as slim


def stem(input):

    with tf.name_scope('Stem'):
        net = slim.conv2d(input, 32, 3, stride = 2, padding = 'VALID', name = 'conv_1_3')
        net = slim.conv2d(net, 32, 3, padding = 'VALID', name = 'conv_2_3')
        net = slim.conv2d(net, 64, 3, name = 'conv_3_3')

        net_0 = slim.max_pool2d(net, 3, stride = 2, padding = 'VALID', name = 'max_pool_4')
        net_1 = slim.conv2d(net, 96, 3, stride = 2, padding = 'VALID', name = 'conv_4_3')
        net = tf.concat([net_0, net_1], axis = 3)

        with tf.name_scope('branch_0'):
            net_0 = slim.conv2d(net, 64, 1, name = 'conv_a_1')
            net_0 = slim.conv2d(net_0, 96, 3, padding = 'VALID', name = 'conv_b_3')
        with tf.name_scope('branch_1'):
            net_1 = slim.conv2d(net, 64, 1, name = 'conv_a_1')
            net_1 = slim.conv2d(net_1, 64, [7, 1], name = 'conv_b_71')
            net_1 = slim.conv2d(net_1, 64, [1, 7], name = 'conv_c_17')
            net_1 = slim.conv2d(net_1, 96, [3, 3], padding = 'VALID', name = 'conv_d_3')
        net = tf.concat([net_0, net_1], axis = 3)

        net_1 = slim.conv2d(net, 192, 3, padding = 'VALID', name = 'conv_6_3')
        net_2 = slim.max_pool2d(net, 3, stride = 2, padding = 'VALID', name = 'max_pool_6')
        return tf.concat([net_1, net_2], axis = 3)


def module_A(net):

    with tf.name_scope('Inception-resnet-A'):
        with tf.name_scope('branch_0'):
            net_0 = slim.conv2d(net, 32, 1, scope = 'conv_a_1')
        with tf.name_scope('branch_1'):
            net_1 = slim.conv2d(net, 32, 1, scope = 'conv_a_1')
            net_1 = slim.conv2d(net_1, 32, 3, scope = 'conv_b_3')
        with tf.name_scope('branch_2'):
            net_2 = slim.conv2d(net, 32, 1, scope = 'conv_a_1')
            net_2 = slim.conv2d(net_2, 48, 3, scope = 'conv_b_3')
            net_2 = slim.conv2d(net_2, 64, 3, scope = 'conv_c_3')

        net_b = tf.concat([net_0, net_1, net_2], axis = 3)
        net_b = slim.conv2d(net_b, 384, 1, scope = 'conv_1')
        net += net_b
        return tf.nn.relu(net)


def reduction_A(net):

    with tf.name_scope('Reduction-A'):
        with tf.name_scope('branch_0'):
            net_0 = slim.max_pool2d(net, 3, stride = 2, padding = 'VALID', scope = 'max_pool_3')
        with tf.name_scope('branch_1'):
            net_1 = slim.conv2d(net, 384, 3, scope = 'conv_a_3')
        with tf.name_scope('branch_2'):
            net_2 = slim.conv2d(net, 256, 1, scope = 'conv_a_1')
            net_2 = slim.conv2d(net_2, 256, 3, scope = 'conv_b_1')
            net_2 = slim.conv2d(net_2, 384, 3, scope = 'conv_c_1')

        return tf.concat([net_0, net_1, net_2], axis = 3)


def module_B(net):

    with tf.name_scope('Inception-resnet-B'):
        with tf.name_scope('branch_0'):
            net_0 = slim.conv2d(net, 192, 1, scope = 'conv_a_1')
        with tf.name_scope('branch_1'):
            net_1 = slim.conv2d(net, 128, 1, scope = 'conv_a_1')
            net_1 = slim.conv2d(net_1, 160, [1, 7], scope = 'conv_b_17')
            net_1 = slim.conv2d(net_1, 192, [7, 1], scope = 'conv_c_71')

        net_b = tf.concat([net_0, net_1], axis = 3)
        net_b = slim.conv2d(net_b, 1154, 1, name = 'conv_1')
        net += net_b
        return tf.nn.relu(net)

def reduction_B(net):

    with tf.name_scope('Reduction-B'):
        with tf.name_scope('branch_0'):
            net_0 = slim.max_pool2d(net, 3, stride = 2, padding = 'VALID', scope = 'max_pool_3')
        with tf.name_scope('branch_1'):
            net_1 = slim.conv2d(net, 256, 1, name = 'conv_a_1')
            net_1 = slim.conv2d(net_1, 384, 3, padding = 'VALID', name = 'conv_b_1')
        with tf.name_scope('branch_2'):
            net_2 = slim.conv2d(net, 256, 1, name = 'conv_a_1')
            net_2 = slim.conv2d(net_2, 288, 3, padding = 'VALID', name = 'conv_b_1')
        with tf.name_scope('branch_3'):
            net_3 = slim.conv2d(net, 256, 1, name = 'conv_a_1')
            net_3 = slim.conv2d(net_3, 288, 3, name = 'conv_b_1')
            net_3 = slim.conv2d(net_3, 320, 3, padding = 'VALID', name = 'conv_c_1')

        return tf.concat([net_0, net_1, net_2, net_3], axis = 3)


def module_C(net):

    with tf.name_scope('Inception-resnet-C'):
        with tf.name_scope('branch_0'):
            net_0 = slim.conv2d(net, 192, 1, scope = 'conv_a_1')
        with tf.name_scope('branch_1'):
            net_1 = slim.conv2d(net, 192, 1, scope = 'conv_a_1')
            net_1 = slim.conv2d(net_1, 224, [1, 3], scope = 'conv_b_13')
            net_1 = slim.conv2d(net_1, 256, [3, 1], scope = 'conv_c_31')

        net_b = tf.concat([net_0, net_1], axis = 3)
        net_b = slim.conv2d(net_b, 2048, 1, name = 'conv_1')
        net += net_b
        return tf.nn.relu(net)


def inception_resnet_v2(input, n_class, decay):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.relu,
                        weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = .01),
                        weights_regularizer = slim.l2_regularizer(decay)):
        with slim.arg_scope([slim.conv2d], padding = 'SAME'):

            net = stem(input)
            net = slim.repeat(net, 5, module_A)
            net = reduction_A(net)
            net = slim.repeat(net, 10, module_B)
            net = reduction_B(net)
            net = slim.repeat(net, 5, module_C)

            net = slim.avg_pool2d(net, 8, stride = 1, padding = 'VALID', name = 'avg_pool')
            net = slim.dropout(net, .8, is_training = True, name = 'dropout')

            return slim.fully_connected(net, n_class, name = 'fc')
