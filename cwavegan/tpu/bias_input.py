# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Read MNIST data as TFRecords and create a tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, os
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('data_file', 'gs://wavegan_tfrecords_int/', 'Training .tfrecord data file')

window_len = 8192
NUM_TRAIN_AUDIO = 60000
NUM_EVAL_AUDIO = 10000


def parser(serialized_example):
  features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
  if True:
    features['label'] = tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
  
  example = tf.parse_single_example(serialized_example, features)
  wav = example['samples']
  label = example['label']

  # first window
  wav = wav[:window_len]

  wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]])

  wav.set_shape([window_len, 1])

  return wav, label


class InputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    mode = ('train' if is_training
            else 'test')
    self.data_file = glob.glob(os.path.join(FLAGS.data_file, mode) + '*.tfrecord')

  def __call__(self, params):
    """Creates a simple Dataset pipeline."""

    batch_size = params['batch_size']

    data_files = []
    for i in range(128):
      data_file = FLAGS.data_file + 'train-{}-of-128.tfrecord'.format(str(i).zfill(3))
      data_files.append(data_file)

    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(parser).cache()
    if self.is_training:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    wav, labels = dataset.make_one_shot_iterator().get_next()

    random_noise = tf.random_uniform([batch_size, self.noise_dim], -1., 1., dtype=tf.float32)
    labels = labels + tf.constant(10, name='fixed', dtype=tf.int64)
    labels = tf.cast(labels, dtype=tf.float32)
    labels = tf.reshape(labels, [batch_size, 1])

    features = {
        'real_audio': wav,
        'random_noise': random_noise}

    return features, labels
