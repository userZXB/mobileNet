import random

import tensorflow as tf
import numpy as np
import os.path
import glob

from tensorflow.python.platform import gfile

from nets import MobileNetV1
slim = tf.contrib.slim
ckpt_path = 'E:/tmp/cotton-fun-model/mobilenet_v1_1.0_224.ckpt'
INPUT_DATA = 'E:/Graduation design/Cotton_photos_Record/cotton_photos'
CACHE_DIR = 'E:/tmp/bottleneck/sm'
TEST_PERCENTAGE = 10
saver = tf.train.Saver()


def mobi_parse_fun(x_in, y_label=0):
    img_path = tf.read_file(x_in)
    img_decode = tf.image.decode_png(img_path, channels=3)
    img = tf.image.resize_images(img_decode, [224, 224])
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img, y_label



X_in = tf.placeholder(tf.string, None)
# Y_in = tf.placeholder(tf.int32, None)
train_data = tf.data.Dataset.from_tensor_slices((X_in))
train_data = train_data.map(mobi_parse_fun)
train_data = train_data.batch(1)
iter_ = tf.data.Iterator.from_structure(train_data.output_types,
                                        train_data.output_shapes)
x_batch, y_batch = iter_.get_next()
train_init_op = iter_.make_initializer(train_data)
with tf.contrib.slim.arg_scope(MobileNetV1.mobilenet_v1_arg_scope()):
    logits, endpoints = MobileNetV1.mobilenet_v1(x_batch, num_classes=7)


img_path = 'E:/Graduation design/Cotton_photos_Record/cotton_photos/gm/*.png'
img_list = glob.glob(img_path)
with tf.Session() as sess:
    saver.restore(sess, ckpt_path)
    ## 查看网络每一层的参数：
    print('print the trainable parameters: ')
    for eval_ in tf.trainable_variables():
        print(eval_.name)
        w_val = sess.run(eval_.name)
        print(w_val.shape)
    sess.run(train_init_op, feed_dict={X_in: img_list})

    # ---------------------------------------------
    # ---------------------------------------------
    # 查看每一层的 feature map，
    key_name = endpoints.keys()
    print('print the feature maps: ')
    for name_ in key_name:
        print(name_)
        feat_map = sess.run(endpoints[name_])
        print(feat_map.shape)
    fc_map = endpoints['AvgPool_1a']
    fc_feat = tf.squeeze(fc_map, [1, 2])
    for img_name in img_list:
        print(img_name)

        x_bat, y_bat = sess.run([x_batch, y_batch])
        print(x_bat.shape, y_bat.shape)

        fc_feature = sess.run([fc_feat])
        print(fc_feature[0].shape)

        break


