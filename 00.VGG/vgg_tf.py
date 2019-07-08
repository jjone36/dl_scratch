# Origial Paper : https://arxiv.org/pdf/1610.02357.pdf

import numpy as np
from keras.utils import to_categorical

import tensorflow as tf
import tf.contrib.slim as slim

# Load the data
def loading_data():

    (X_tr, y_tr), (X_val, y_val) = keras.datasets.cifar100()

    X_tr = X_tr.astype('float32') / 255
    y_tr = to_categorical(y_tr, num_classes= 1000)

    X_val = X_val.astype('float32') / 255
    y_val = to_categorical(y_val, num_classes= 1000)

    return (X_tr, y_tr), (X_val, y_val)


def vgg16(input, n_class, decay = 5*1e-4):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.relu,
                        weights_initializer = tf.truncated_normal_initializer(mean = 0, stddev = .01),
                        weights_regularizer = slim.l2_regularizer(decay)):
        with slim.arg_scope([slim.conv2d], padding = 'SAME'):

            net = slim.repeat(input, repetition = 2, layer = slim.conv2d, 64, [3, 3], scope = 'conv_1')
            net = slim.max_pool2d(net, [2, 2], stride = 2, scope = 'pool1')

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope = 'conv_2')
            net = slim.max_pool2d(net, [2, 2], stride = 2, scope = 'pool2')

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope = 'conv_3')
            net = slim.max_pool2d(net, [2, 2], stride = 2, scope = 'pool3')

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv_4')
            net = slim.max_pool2d(net, [2, 2], stride = 2, scope = 'pool4')

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope = 'conv_5')
            net = slim.max_pool2d(net, [2, 2], stride = 2, scope = 'pool5')

            net = slim.fully_connected(net, 4096, scope = 'fc6')
            net = slim.dropout(net, .5, scope = 'dropout6')

            net = slim.fully_connected(net, 4096, scope = 'fc7')
            net = slim.dropout(net, .5, scope = 'dropout7')

            net = slim.fully_connected(net, n_class, activation_fn = None, scope = 'fc8')

    return net


# Mini-batch
def create_batch(X_tr, y_tr, batch_size):

    m = X_tr.shape[0]
    NUM = list(np.random.permutation(m))
    X_shuffled = X_train[NUM, :]
    y_shuffled = y_train[NUM, :]

    n_batch = int(m/batch_size)
    batches = []

    for i in range(0, n_batch):
        X_batch = X_shuffled[i*batch_size:(i+1)*batch_size, :, :, :]
        y_batch = y_shuffled[i*batch_size:(i+1)*batch_size, :]
        batch = (X_batch, y_batch)
        batches.append(batch)

    # Tail of the batches
    X_batch_end = X_shuffled[n_batch*batch_size+1:, :, :, :]
    y_batch_end = y_shuffled[n_batch*batch_size+1:, :]
    batch = (X_batch_end, y_batch_end)
    batches.append(batch)

    return batches

# Cost funtion
def compute_cost(pred, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
    return loss


# Plot the cost
def plot_cost(costs, pred, y):

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    pred_op = tf.argmax(pred, 1)
    actual = tf.argmax(y, 1)
    correct_pred = tf.equal(pred_op, actual)
    acc = tf.reduce_mean(tf.cast(correct_pred, 'float'))
    return acc


def main():

    batch_size = 256
    epochs = 5

    (X_tr, y_tr), (X_val, y_val) = loading_data()

    (m, im_size, im_size, 3) = X_tr.shape
    n_class = y_tr.shape[1]
    costs = []

    X = tf.placeholder(tf.float32, [None, im_size, im_size, 3])
    y = tf.placeholder(tf.float32, [None, n_class])

    # Create a graph
    pred = vgg16(X, n_class)

    loss = compute_cost(pred, y)
    optimizer = tf.train.RMSPropOptimizer(lr = .01, decay=0.99, momentum=0.9).minimize(loss)

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epochs):

            batches = create_batch(X_tr, y_tr, batch_size=batch_size)
            n_batch = int(m/batch_size)

            batch_cost = 0
            for batch in batches:
                X_batch, y_batch = batch
                _, temp_cost = sess.run([optimizer, loss], feed_dict = {X : X_batch, y: y_batch})
                batch_cost += temp_cost / n_batch

            # Print the cost per each epoch
            if epoch % 10 == 0:
                print("Cost after {0} epoch: {1}".format(epoch, batch_cost))
            if epoch % 1 == 0:
                costs.append(batch_cost)

    # step 7. plot the cost
    acc = plot_cost(costs, pred, y_tr)
    print(acc)


if __name__=="__main__":
    main()
