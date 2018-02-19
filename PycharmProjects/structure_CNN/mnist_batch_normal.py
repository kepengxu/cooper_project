#
#   my_nn_lib.py
#       date. 5/19/2016
#

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys
import numpy as np
# import cv2
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
#chkpt_file = '../MNIST_data/mnist_cnn.ckpt'



# Convolution 2-D Layer
class Convolution2D(object):
    '''
      constructor's args:
          input     : input image (2D matrix)
          input_siz ; input image size
          in_ch     : number of incoming image channel
          out_ch    : number of outgoing image channel
          patch_siz : filter(patch) size
          weights   : (if input) (weights, bias)
    '''

    def __init__(self, input, input_siz, in_ch, out_ch, patch_siz, activation='relu'):#default is relu activate net
        self.input = input
        self.rows = input_siz[0]
        self.cols = input_siz[1]
        self.in_ch = in_ch
        self.activation = activation

        wshape = [patch_siz[0], patch_siz[1], in_ch, out_ch]

        w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1),
                           trainable=True)
        b_cv = tf.Variable(tf.constant(0.1, shape=[out_ch]),
                           trainable=True)

        self.w = w_cv
        self.b = b_cv
        self.params = [self.w, self.b]

    def output(self):
        shape4D = [-1, self.rows, self.cols, self.in_ch]

        x_image = tf.reshape(self.input, shape4D)  # reshape to 4D tensor
        linout = tf.nn.conv2d(x_image, self.w,
                              strides=[1, 1, 1, 1], padding='SAME') + self.b
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        else:
            self.output = linout

        return self.output


# Max Pooling Layer
class MaxPooling2D(object):
    '''
      constructor's args:
          input  : input image (2D matrix)
          ksize  : pooling patch size
    '''

    def __init__(self, input, ksize=None):
        self.input = input
        if ksize == None:
            ksize = [1, 2, 2, 1]
            self.ksize = ksize

    def output(self):
        self.output = tf.nn.max_pool(self.input, ksize=self.ksize,
                                     strides=[1, 2, 2, 1], padding='SAME')

        return self.output


# Full-connected Layer
class FullConnected(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_h = tf.Variable(tf.truncated_normal([n_in, n_out],
                                              mean=0.0, stddev=0.05), trainable=True)
        b_h = tf.Variable(tf.zeros([n_out]), trainable=True)

        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)

        return self.output


# Read-out Layer
class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input

        w_o = tf.Variable(tf.random_normal([n_in, n_out],
                                           mean=0.0, stddev=0.05), trainable=True)
        b_o = tf.Variable(tf.zeros([n_out]), trainable=True)

        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]

    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.softmax(linarg)

        return self.output


#


#
#  mnist_cnn_bn.py   date. 5/21/2016
#                    date. 6/2/2017 check TF 1.1 compatibility
#



def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


#

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return accuracy


def mlogloss(predicted, actual):
    '''
      args.
         predicted : predicted probability
                    (sum of predicted proba should be 1.0)
         actual    : actual value, label
    '''

    def inner_fn(item):
        eps = 1.e-15
        item1 = min(item, (1 - eps))
        item1 = max(item, eps)
        res = np.log(item1)

        return res

    nrow = actual.shape[0]
    ncol = actual.shape[1]

    mysum = sum([actual[i, j] * inner_fn(predicted[i, j])
                 for i in range(nrow) for j in range(ncol)])

    ans = -1 * mysum / nrow

    return ans


#

# Create the model
def inference(x, y_, keep_prob, phase_train):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope('conv_1'):
        conv1 = Convolution2D(x, (28, 28), 1, 32, (5, 5), activation='none')
        conv1_bn = batch_norm(conv1.output(), 32, phase_train)
        conv1_out = tf.nn.relu(conv1_bn)

        pool1 = MaxPooling2D(conv1_out)
        pool1_out = pool1.output()

    with tf.variable_scope('conv_2'):
        conv2 = Convolution2D(pool1_out, (28, 28), 32, 64, (5, 5),
                              activation='none')
        conv2_bn = batch_norm(conv2.output(), 64, phase_train)
        conv2_out = tf.nn.relu(conv2_bn)

        pool2 = MaxPooling2D(conv2_out)
        pool2_out = pool2.output()
        pool2_flat = tf.reshape(pool2_out, [-1, 7 * 7 * 64])

    with tf.variable_scope('fc1'):
        fc1 = FullConnected(pool2_flat, 7 * 7 * 64, 1024)
        fc1_out = fc1.output()
        fc1_dropped = tf.nn.dropout(fc1_out, keep_prob)

    y_pred = ReadOutLayer(fc1_dropped, 1024, 10).output()

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_pred),
                                                  reduction_indices=[1]))
    loss = cross_entropy
    train_step = training(loss, 1.e-4)
    accuracy = evaluation(y_pred, y_)

    return loss, accuracy, y_pred


#
if __name__ == '__main__':
    TASK = 'train'  # 'train' or 'test'

    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    loss, accuracy, y_pred = inference(x, y_,
                                       keep_prob, phase_train)

    # Train
    lr = 0.01
    train_step = tf.train.AdagradOptimizer(lr).minimize(loss)
    vars_to_train = tf.trainable_variables()  # option-1
    vars_for_bn1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  # TF >1.0
                                     scope='conv_1/bn')
    vars_for_bn2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  # TF >1.0
                                     scope='conv_2/bn')
    vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
    vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))

    if TASK == 'test' or os.path.exists(chkpt_file):
        restore_call = True
        vars_all = tf.all_variables()
        vars_to_init = list(set(vars_all) - set(vars_to_train))
        init = tf.variables_initializer(vars_to_init)  # TF >1.0
    elif TASK == 'train':
        restore_call = False
        init = tf.global_variables_initializer()  # TF >1.0
    else:
        print('Check task switch.')

    saver = tf.train.Saver(vars_to_train)  # option-1
    # saver = tf.train.Saver()                   # option-2


    with tf.Session() as sess:
        # if TASK == 'train':              # add in option-2 case
        sess.run(init)  # option-1

        if restore_call:
            # Restore variables from disk.
            saver.restore(sess, chkpt_file)

        if TASK == 'train':
            print('\n Training...')
            for i in range(5001):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.5,
                                phase_train: True})
                if i % 1000 == 0:
                    cv_fd = {x: batch_xs, y_: batch_ys, keep_prob: 1.0,
                             phase_train: False}
                    train_loss = loss.eval(cv_fd)
                    train_accuracy = accuracy.eval(cv_fd)

                    print('  step, loss, accurary = %6d: %8.4f, %8.4f' % (i,
                                                                          train_loss, train_accuracy))

        # Test trained model
        test_fd = {x: mnist.test.images, y_: mnist.test.labels,
                   keep_prob: 1.0, phase_train: False}
        print(' accuracy = %8.4f' % accuracy.eval(test_fd))
        # Multiclass Log Loss
        pred = y_pred.eval(test_fd)
        act = mnist.test.labels
        print(' multiclass logloss = %8.4f' % mlogloss(pred, act))

        # Save the variables to disk.
        if TASK == 'train':
            save_path = saver.save(sess, chkpt_file)
            print("Model saved in file: %s" % save_path)
