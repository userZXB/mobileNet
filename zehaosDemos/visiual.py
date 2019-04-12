
# coding=utf-8#

"""孪生卷积神经网络"""

import tensorflow as tf
import numpy as np
import math


class SiameseNet(object):

    def __init__(self):
        print("正在构建孪生网络...")
        self.opts = {'trainWeightDecay': 0.0, 'stddev': 0.01}

    def build_siamese_cnn_network(self, input):
        first_image = tf.expand_dims(input[:, :, :, 0], 3)
        second_image = tf.expand_dims(input[:, :, :, 1], 3)

        with tf.variable_scope('siamese_CNN') as scope:
            first_output = self._build_branch(first_image, True)
            scope.reuse_variables()
            second_output = self._build_branch(second_image, False)

        return first_output, second_output

    def _build_branch(self, image, branch):
        print("构建孪生网络分支...")

        with tf.variable_scope('conv_block1_1'):
            print("构建conv1,relu1...")
            name = tf.get_variable_scope().name
            outputs = self._conv(image, 64, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('conv_block1_2'):
            print("构建conv2,relu2,max-pooling1...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 64, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)
            outputs = self._maxPool(outputs, 2, 2)
            conv_block1 = outputs

        with tf.variable_scope('conv_block2_1'):
            print("构建conv3,relu3...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 64, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('conv_block2_2'):
            print("构建,conv4,relu4,max-pooling2...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 64, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)
            outputs = self._maxPool(outputs, 2, 2)
            conv_block2 = outputs

        with tf.variable_scope('conv_block3_1'):
            print("构建conv5,relu5...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 128, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('conv_block3_2'):
            print("构建conv6,relu6,max-pooling3...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 128, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)
            outputs = self._maxPool(outputs, 2, 2)
            conv_block3 = outputs

        with tf.variable_scope('conv_block4_1'):
            print("构建conv7,relu7...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 128, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('conv_block4_2'):
            print("构建conv8,relu8,max-pooling4...")
            name = tf.get_variable_scope().name
            outputs = self._conv(outputs, 128, 3, 1, self.opts['trainWeightDecay'], self.opts['stddev'])
            outputs = tf.nn.relu(outputs)

        if branch:
            self.conv_block1_1 = conv_block1
            self.conv_block1_2 = conv_block2
            self.conv_block1_3 = conv_block3
            self.conv_block1_4 = outputs
        else:
            self.conv_block2_1 = conv_block1
            self.conv_block2_2 = conv_block2
            self.conv_block2_3 = conv_block3
            self.conv_block2_4 = outputs

        return outputs

    def _conv(self, inputs, filters, size, stride, wd, stddev, name=None):
        channels = int(inputs.get_shape()[-1])

        with tf.variable_scope('conv'):
            weights = self.getVariable('weights', shape=[size, size, channels, filters],
                                       initializer=tf.random_normal_initializer(),
                                       weightDecay=wd, dType=tf.float32, trainable=True)
            biases = self.getVariable('biases', shape=[filters, ],
                                      initializer=tf.constant_initializer(dtype=tf.float32),
                                      weightDecay=0.0, dType=tf.float32, trainable=True)

        p = np.floor((size - 1) / 2).astype(np.int32)
        p_x = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]])
        conv = tf.nn.conv2d(p_x, weights, strides=[1, stride, stride, 1], padding='VALID')
        if name is not None:
            conv = tf.add(conv, biases, name=name)
        else:
            conv = tf.add(conv, biases)

        print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
            size, size, stride, filters, channels))
        return conv

    def _maxPool(self, inputs, kSize, _stride):
        with tf.variable_scope('poll'):
            p = np.floor((kSize - 1) / 2).astype(np.int32)
            p_x = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]])
            output = tf.nn.max_pool(p_x, ksize=[1, kSize, kSize, 1], strides=[1, _stride, _stride, 1],
                                    padding='VALID')
        return output

    def getVariable(self, name, shape, initializer, weightDecay=0.0, dType=tf.float32, trainable=True):
        if weightDecay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
        else:
            regularizer = None

        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dType, regularizer=regularizer,
                               trainable=trainable)

    def build_summaries(self):
        with tf.name_scope('CNN_outputs'):
            tf.summary.image('conv_block1_1_1', self._concact_features(self.conv_block1_1[:, :, :, 0:16]), 1)  # 取部分特征图显示
            tf.summary.image('conv_block1_1_2', self._concact_features(self.conv_block1_1[:, :, :, 16:32]), 1)
            tf.summary.image('conv_block1_1_3', self._concact_features(self.conv_block1_1[:, :, :, 32:48]), 1)
            tf.summary.image('conv_block1_1_4', self._concact_features(self.conv_block1_1[:, :, :, 48:64]), 1)
            tf.summary.image('conv_block1_2', self._concact_features(self.conv_block1_2), 1)
            tf.summary.image('conv_block1_3', self._concact_features(self.conv_block1_3), 1)
            tf.summary.image('conv_block1_4', self._concact_features(self.conv_block1_4), 1)
            tf.summary.image('conv_block2_1', self._concact_features(self.conv_block2_1), 1)
            tf.summary.image('conv_block2_2', self._concact_features(self.conv_block2_2), 1)
            tf.summary.image('conv_block2_3', self._concact_features(self.conv_block2_3), 1)
            tf.summary.image('conv_block2_4', self._concact_features(self.conv_block2_4), 1)

    def _concact_features(self, conv_output):
        """
        对特征图进行reshape拼接
        :param conv_output:输入多通道的特征图
        :return:
        """
        # 取出每一层卷积核的个数
        num_or_size_splits = conv_output.get_shape().as_list()[-1]
        # 进行切割，取得每一个卷积核上的特征
        each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
        # 组成一个正方形矩阵，按照其平方数进行组合
        concact_size = int(math.sqrt(num_or_size_splits) / 1)
        all_concact = None
        for i in range(concact_size):
            row_concact = each_convs[i * concact_size]
            for j in range(concact_size - 1):
                row_concact = tf.concat([row_concact, each_convs[i * concact_size + j + 1]], 1)
            if i == 0:
                all_concact = row_concact
            else:
                all_concact = tf.concat([all_concact, row_concact], 2)

        return all_concact
