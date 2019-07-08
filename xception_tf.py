# Origial Paper : https://arxiv.org/pdf/1610.02357.pdf

import numpy as np
from keras.utils import to_categorical

import tensorflow as tf

# Load the data
def loading_data():

    (X_tr, y_tr), (X_val, y_val) = keras.datasets.cifar100()

    X_tr = X_tr.astype('float32') / 255
    y_tr = to_categorical(y_tr, num_classes= 1000)

    X_val = X_val.astype('float32') / 255
    y_val = to_categorical(y_val, num_classes= 1000)

    return (X_tr, y_tr), (X_val, y_val)


def middle_block(X):
    temp = X
    # Main branch
    X = tf.nn.relu(X)
    X = tf.nn.separable_conv2d(X, 728, [3, 3])
    X = tf.nn.batch_normalization(X)

    X = tf.nn.relu(X)
    X = tf.nn.separable_conv2d(X, 728, [3, 3])
    X = tf.nn.batch_normalization(X)

    X = tf.nn.relu(X)
    X = tf.nn.separable_conv2d(X, 728, [3, 3])
    X = tf.nn.batch_normalization(X)

    return tf.add(X, temp)


def xception(input, n_middle_block = 8, n_class):

    ######### Entry Flow
    X = tf.layers.conv2d(inputs = input, filters = 32, kernel_size = 3, strides = 2)
    X = tf.nn.batch_normalization(X)
    X = tf.nn.relu(X)

    X = tf.layers.conv2d(inputs = X, filters = 64, kernel_size = 3)
    X = tf.nn.batch_normalization(X)

    for i in [128, 256, 728]:
        temp = X
        # Main branch
        X = tf.nn.relu(X)
        X = tf.nn.separable_conv2d(X, i, [3, 3])
        X = tf.nn.batch_normalization(X)

        X = tf.nn.relu(X)

        X = tf.nn.separable_conv2d(X, i, [3, 3])
        X = tf.nn.batch_normalization(X)

        X = tf.nn.max_pool(X, ksize = 3, strides = 2, padding = 'SAME')
        # Side branch
        X_s = tf.layers.conv2d(temp, filters = i, kernel_size = 1, strides = 2)
        X_s = tf.nn.batch_normalization(X_s)

        X = tf.add(X, X_s)

    ######### Middle Flow
    for i in range(n_middle_block):
        X = middle_block(X)

    ######### Exit Flow
    temp = X

    # Main branch
    X = tf.nn.relu(X)
    X = tf.nn.separable_conv2d(X, 728, [3, 3])
    X = tf.nn.batch_normalization(X)

    X = tf.nn.relu(X)
    X = tf.nn.separable_conv2d(X, 1024, [3, 3])
    X = tf.nn.batch_normalization(X)

    X = tf.nn.max_pool(X, ksize = 3, strides = 2, padding = 'SAME')

    # Side branch
    X_s = tf.layers.conv2d(temp, filters = 1024, kernel_size = 1, strides = 2)

    X = tf.add(X, X_s)

    X = tf.nn.separable_conv2d(X, 1536, [3, 3])
    X = tf.nn.batch_normalization(X)
    X = tf.nn.relu(X)

    X = tf.nn.separable_conv2d(X, 2048, [3, 3])
    X = tf.nn.batch_normalization(X)
    X = tf.nn.relu(X)

    X = tf.reduce_mean(X, axis = [1, 2])
    X = tf.layers.dense(X, units = n_class)

    return X


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
def compute_cost(pred, y_val):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y_val))
    return loss


# Plot the cost
def plot_cost(costs, y_hat, y):

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    pred_op = tf.argmax(y_hat, 1)
    actual = tf.argmax(y, 1)
    correct_pred = tf.equal(pred_op, actual)
    acc = tf.reduce_mean(tf.cast(correct_pred, 'float'))
    return acc

def main():

    lr = .001
    batch_size = 10
    epochs = 5

    (X_tr, y_tr), (X_val, y_val) = loading_data()

    (m, im_size, im_size, 3) = X_tr.shape
    n_class = y_tr.shape[1]
    costs = []

    X = tf.placeholder(tf.float32, [None, im_size, im_size, 3])
    y = tf.placeholder(tf.float32, [None, n_class])


    # Create a graph
    pred = xception(input = X, n_class)

    loss = compute_cost(pred, y)
    optimizer = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(loss)

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
    acc = plot_cost(costs, y_hat = Z3, y = y_train)
    print(acc)


if __name__=="__main__":
    main()
