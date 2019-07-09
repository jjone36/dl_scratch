# Origial Paper : https://arxiv.org/pdf/1409.4842.pdf

import numpy as np

import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, concatenate
from keras.layers import Input
from keras.layers import AveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler


# Load the data
def loading_data():

    (X_tr, y_tr), (X_val, y_val)   # Load the dataset

    X_tr = X_tr.astype('float32') / 255
    y_tr = to_categorical(y_tr, num_classes= 1000)

    X_val = X_val.astype('float32') / 255
    y_val = to_categorical(y_val, num_classes= 1000)

    return (X_tr, y_tr), (X_val, y_val)


# Define the inception module
def inception_module(X, fm_sizes, kernel_init, bias_init):
    """
    Inception V1 Module - sparsely connected structure
    input - X : input data
            fm_sizes : the number of filters for each convolution
                     following the order of (#1x1, #3x3_reduce, #3x3, #5x5_reduce, #5x5, #proj)
            kernel_init, bias_init : the weights initialization
    output - the outcome of an inception module
    """
    # 1x1 conv
    X_1 = Conv2D(filters= fm_sizes[0], kernel_size=1, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X)

    # 1x1 conv - 3x3 conv
    X_3 = Conv2D(filters= fm_sizes[1], kernel_size=1, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X)
    X_3 = Conv2D(filters= fm_sizes[2], kernel_size=1, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X_3)

    # 3x3 conv - 5x5 conv
    X_5 = Conv2D(filters= fm_sizes[3], kernel_size=1, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X)
    X_5 = Conv2D(filters= fm_sizes[4], kernel_size=1, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X_5)

    # max pool - 1x1 conv
    X_p = MaxPool2D(pool_size=(3, 3), strides=1, padding='SAME')(X)
    X_p = Conv2D(filters= fm_sizes[5], kernel_size=1, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X_p)

    # concatenate
    output = concatenate([X_1, X_3, X_5, X_p], axis = 3)
    return output


def side_branch(X, kernel_init, bias_init):
    """
    Inception V1 Side Branch
    input - X : input data
            kernel_init, bias_init : the weights initialization
    output - the outcome of the side branch
    """
    # average pool - 5x5 conv, s = 3
    X = AveragePooling2D(pool_size=(5, 5), strides= 3)(X)
    # 1x1 conv
    X = Conv2D(filters= 128, kernel_size=1, padding = 'SAME', activation = 'relu',
              kernel_initializer= kernel_init, bias_initializer= bias_init)(X)
    # fully connected layer
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    # Dropout
    X = Dropout(.7)(X)
    # softmax
    output = Dense(units = 1000, activation = 'softmax')(X)
    return output


def inception(input_shape, kernel_init, bias_init):
    """
    Build Inception Model
    input - input_shape = (None, 224, 224, 3)
            kernel_init, bias_init : weight initialization
    output- Compiled Inception Network
    """

    # input layer
    intput = Input(shape = input_shape)
    # 7x7 conv, s = 2
    X = Conv2D(64, kernel_size=(7, 7), strides = 2, padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(input)
    # max pool, s = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2, padding = 'SAME')(X)
    # 1x1 conv - 3x3 conv
    X = Conv2D(64, kernel_size=(1, 1), padding = 'SAME', padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X)
    X = Conv2D(192, kernel_size=(3, 3), padding = 'SAME', padding = 'SAME', activation = 'relu',
                kernel_initializer= kernel_init, bias_initializer= bias_init)(X)
    # max pool, s = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2, padding = 'SAME')(X)

    # incetpion_module 3a, 3b
    X = inception_module(X, fm_sizes=[64, 96, 128, 16, 32, 32], kernel_init, bias_init)(X)
    X = inception_module(X, fm_sizes=[128, 128, 192, 32, 96, 64], kernel_init, bias_init)(X)

    # max pool, s = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2, padding = 'SAME')(X)

    # incetpion_module 4a
    X = inception_module(X, fm_sizes=[192, 96, 208, 16, 48, 64], kernel_init, bias_init)(X)
    # Side branch 4a
    X_s1 = side_branch(X, kernel_init, bias_init)

    # incetpion_module 4b, 4c, 4d
    X = inception_module(X, fm_sizes=[160, 112, 224, 24, 64, 64], kernel_init, bias_init)(X)
    X = inception_module(X, fm_sizes=[128, 128, 256, 24, 64, 64], kernel_init, bias_init)(X)
    X = inception_module(X, fm_sizes=[112, 144, 288, 32, 64, 64], kernel_init, bias_init)(X)

    # incetpion_module 4e
    X = inception_module(X, fm_sizes=[256, 160, 320, 32, 128, 128], kernel_init, bias_init)(X)
    # Side branch 4e
    X_s2 = side_branch(X, kernel_init, bias_init)

    # max pool, s = 2
    X = MaxPool2D(pool_size=(3, 3), strides=2, padding = 'SAME')(X)

    # incetpion_module 5a, 5b
    X = inception_module(X, fm_sizes=[256, 160, 320, 32, 128, 128], kernel_init, bias_init)(X)
    X = inception_module(X, fm_sizes=[384, 192, 384, 48, 128, 128], kernel_init, bias_init)(X)

    # global average pool -> 1x1
    X = GlobalMaxPooling2D()(X)
    # dropout
    X = Dropout(.4)(X)
    # softmax
    output = Dense(1000, activation='softmax')(X)

    return Model(inputs = input, outputs = [output, X_s1, X_s2])


def decay(epoch, init_lr):
    learning_rate = init_lr
    ratio = 0.96
    epoch_step = 8
    drop = np.power(ratio, np.floor((epoch+1)/epoch_step))
    return init_lr * drop

# initialization
epochs = 25
init_lr = .01

if __name__=='__main__':

    bias_init = keras.initializers.Constant(.2)
    kernel_init = keras.initializers.glorot_uniform()

    model = inception(input_shape=(None, 224, 224, 3), kernel_init, bias_init)
    model.compile(optimizers = SGD(lr = init_lr, momentum= .9),
                metrics = ['accuracy'],
                # since outputs = [output, X_s1, X_s2]
                loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                loss_weights = [1, .3, .3])

    lr_sc = LearningRateScheduler(schedule=decay(init_lr))

    (X_tr, y_tr), (X_val, y_val) = loading_data()

    model.fit(X_tr, [y_tr, y_tr, y_tr],
            batch_size= 64,
            epochs=epochs,
            callbacks= [lr_sc],
            validation_data= (X_val, [y_val, y_val, y_val]))
