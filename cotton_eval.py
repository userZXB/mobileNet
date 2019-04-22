# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import os
import shutil

import re
import tensorflow as tf
import time

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
model_path = "E:/model"
EVAL_INTERVAL_SECS = 1200
# batch_size需要修改，毕竟测试数据都没有那么多
tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
# 这个也需要修改，因为不是保存在这个路径下的
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'E:/tmp/cotton-fun-model/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'E:/tmp/cotton-eval-model/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cottons', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'E:/Graduation design/Cotton_photos_Record', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.99,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_float('width_multiplier', 1.0,
                          'Width Multiplier, for MobileNet only.')

FLAGS = tf.app.flags.FLAGS
root = "E:/tmp/cotton-fun-model/"
Dir = "E:/model/test/"


def copy_checkpoint():
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            all_files.append(filename)
    begin = all_files[-1].index('-')
    # 取出最后一个模型的轮数
    end = all_files[-1].index(".meta")
    name = all_files[-1][begin+1:end]
    targetDir = Dir + name
    # 如果这个模型不存在
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        for file in all_files:
            file_path = os.path.join(root, file)
            if file == 'checkpoint' or file == 'graph.pbtxt':
                shutil.copy(file_path, targetDir)
            else:
                str1 = 'model.ckpt-'+name+".*"
                match_obj = re.match(str1, file)
                if match_obj:
                    shutil.copy(file_path, targetDir)
    return True


def do_nothing():
    return False


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            'cottons', 'train', 'E:/Graduation Design/Cotton_photos_Record')
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False,
            width_multiplier=FLAGS.width_multiplier)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Precision': slim.metrics.streaming_precision(predictions, labels),
            'recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5)
        })
        # Print the summaries to screen.
        for name, value in names_to_values.items():
            # if name == 'Accuracy':
            #     b = tf.constant(0.8)
            #     result = tf.cond(tf.less(b, value), copy_checkpoint, do_nothing)
            # 先在指定目录下生成以准确率命名的文件夹，获取最新断点模型文件，写入新生成的文件
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        while True:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path

            tf.logging.info('Evaluating %s' % checkpoint_path)
            slim.evaluation.evaluate_once(
                master=FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore)
            time.sleep(EVAL_INTERVAL_SECS)


if __name__ == '__main__':
    tf.app.run()
