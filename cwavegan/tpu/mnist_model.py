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
"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
  return tf.layers.dense(
      x, channels,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def _conv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def discriminator(x, is_training=True, scope='Discriminator'):
  # conv64-lrelu + conv128-bn-lrelu + fc1024-bn-lrelu + fc1
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = _conv2d(x, 64, 4, 2, name='d_conv1')
    x = _leaky_relu(x)

    x = _conv2d(x, 128, 4, 2, name='d_conv2')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

    x = tf.reshape(x, [-1, 7 * 7 * 128])

    x = _dense(x, 1024, name='d_fc3')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

    x = _dense(x, 1, name='d_fc4')

    return x

def discriminator1(x, is_training=True, scope='Discriminator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # initializer
    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
    b_init = tf.constant_initializer(0.0)

    # 1st hidden layer
    conv1 = tf.layers.conv2d(x, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
    lrelu1 = lrelu(conv1, 0.2)

    # 2nd hidden layer
    conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
    lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_training), 0.2)

    # output layer
    conv3 = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
    
    return conv3

def generator(x, is_training=True, scope='Generator'):
  # fc1024-bn-relu + fc6272-bn-relu + deconv64-bn-relu + deconv1-tanh
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = _dense(x, 1024, name='g_fc1')
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))

    x = _dense(x, 7 * 7 * 128, name='g_fc2')
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))

    x = tf.reshape(x, [-1, 7, 7, 128])

    x = _deconv2d(x, 64, 4, 2, name='g_dconv3')
    x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))

    x = _deconv2d(x, 1, 4, 2, name='g_dconv4')
    x = tf.tanh(x)

    return x

# TODO(chrisying): objective score (e.g. MNIST score)

