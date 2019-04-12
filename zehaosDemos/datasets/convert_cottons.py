# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
# 这个脚本的作用是下载flowers_image并把数据集整理成TFRecord的格式
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
# _DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.350
_NUM_VALIDATION = 595

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
# 每个数据集分成五个tfrecord文件 5
_NUM_SHARDS = 9
sub_filename = 'cotton_photos'
tfrecord_filename = 'cottons_%s_%05d-of-%05d.tfrecord'
dir_path =  "E:\Graduation design\Cotton_photos_Record\\bestImage"

# 通道数是3的png文件解码


# 得到图片的维度信息
def read_image_dims(sess, image_data):
    image = decode_png(sess, image_data)
    return image.shape[0], image.shape[1]


def decode_png( sess, image_data):
    _decode_png_data = tf.placeholder(dtype=tf.string)
    _decode_png = tf.image.decode_png(_decode_png_data, channels=3)
    image = sess.run(_decode_png,
                     feed_dict={_decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    # 得到图片的路径目录下的子目录
    flower_root = os.path.join(dataset_dir, sub_filename)
    directories = []  # 存放子目录路径
    class_names = []  # 存放子目录文件名
    for filename in os.listdir(flower_root):
        path = os.path.join(flower_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []  # 存放每一张图片的路径
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
    # 把每一张图片的路径信息与子文件夹名字保存
    return photo_filenames, sorted(class_names)


# 按照一定格式得到数据集的路径，其中split_name指的是training,或者validation
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = tfrecord_filename % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    # 断言
    assert split_name in ['train', 'validation']

    # 每一个shard的图片数量
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                # 感觉这里有点毛病，split_name是一个数组，如何确保单项相加
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()
                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = read_image_dims(sess, image_data)
                        # 得到每一张图片的目录，即子文件的名称
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        example = dataset_utils.image_to_tfexample(
                            image_data, b'png', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


# 把原始的图片文件处理掉
# def _clean_up_temporary_files(dataset_dir):
#   """Removes temporary files used to create the dataset.
#
#   Args:
#     dataset_dir: The directory where the temporary files are stored.
#   """
#   filename = _DATA_URL.split('/')[-1]
#   filepath = os.path.join(dataset_dir, filename)
#   tf.gfile.Remove(filepath)
#
#   tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#   tf.gfile.DeleteRecursively(tmp_dir)

# 判断tfRecord文件夹是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


if __name__ == '__main__':
  dataset_dir = dir_path
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
  else:
    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir) 不用下载
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    # 每一类别都转换成一个id
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)# 其实已经打乱了顺序
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    # _clean_up_temporary_files(dataset_dir+"flowers_photos")
    print('\nFinished converting the Cottons dataset!')

