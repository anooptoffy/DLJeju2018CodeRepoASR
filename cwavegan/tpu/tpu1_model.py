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

def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
    if upsample == 'zeros':
        return tf.layers.conv2d_transpose(
            tf.expand_dims(inputs, axis=1),
            filters,
            (1, kernel_width),
            strides=(1, stride),
            padding='same'
        )[:, 0]
    else:
        raise NotImplementedError


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x

# def _leaky_relu(x):
#   return tf.nn.leaky_relu(x, alpha=0.2)

# leaky_relu
# def lrelu(X, leak=0.2):
#     f1 = 0.5 * (1 + leak)
#     f2 = 0.5 * (1 - leak)
#     return f1 * X + f2 * tf.abs(X)


# def _batch_norm(x, is_training, name):
#   return tf.layers.batch_normalization(
#       x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)
#
#
# def _dense(x, channels, name):
#   return tf.layers.dense(
#       x, channels,
#       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#       name=name)
#
#
# def _conv2d(x, filters, kernel_size, stride, name):
#   return tf.layers.conv2d(
#       x, filters, [kernel_size, kernel_size],
#       strides=[stride, stride], padding='same',
#       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#       name=name)
#
#
# def _deconv2d(x, filters, kernel_size, stride, name):
#   return tf.layers.conv2d_transpose(
#       x, filters, [kernel_size, kernel_size],
#       strides=[stride, stride], padding='same',
#       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
#       name=name)


# def discriminator(x, is_training=True, scope='Discriminator'):
#   # conv64-lrelu + conv128-bn-lrelu + fc1024-bn-lrelu + fc1
#   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#     x = _conv2d(x, 64, 4, 2, name='d_conv1')
#     x = _leaky_relu(x)
#
#     x = _conv2d(x, 128, 4, 2, name='d_conv2')
#     x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))
#
#     x = tf.reshape(x, [-1, 7 * 7 * 128])
#
#     x = _dense(x, 1024, name='d_fc3')
#     x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))
#
#     x = _dense(x, 1, name='d_fc4')
#
#     return x

# def discriminator1(x, is_training=True, scope='Discriminator'):
#   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#     # initializer
#     w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
#     b_init = tf.constant_initializer(0.0)
#
#     # 1st hidden layer
#     conv1 = tf.layers.conv2d(x, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
#     lrelu1 = lrelu(conv1, 0.2)
#
#     # 2nd hidden layer
#     conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
#     lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=is_training), 0.2)
#
#     # output layer
#     conv3 = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
#
#     return conv3

"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def discriminator_wavegan(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=True,
    phaseshuffle_rad=0,
    reuse=False,
    scope='Discriminator'):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(x)[0]

        if use_batchnorm:
            batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
        else:
            batchnorm = lambda x: x

        if phaseshuffle_rad > 0:
            phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
        else:
            phaseshuffle = lambda x: x

        with tf.variable_scope('discriminator_0', reuse=reuse):
            # Layer 0
            # [16384, 1] -> [4096, 64]
            output = x
            output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME', name='downconv_0')
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 1
            # [4096, 64] -> [1024, 128]
            output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME', name='downconv_1')
            output = batchnorm(output)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 2
            # [1024, 128] -> [256, 256]
            output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME', name='downconv_2')
            output = batchnorm(output)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 3
            # [256, 256] -> [64, 512]
            output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME', name='downconv_3')
            output = batchnorm(output)
            output = lrelu(output)
            output = phaseshuffle(output)

            # Layer 4
            # [64, 512] -> [16, 1024]
            output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME', name='downconv_4')
            output = batchnorm(output)
            output = lrelu(output)

        # Flatten
        output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])

        # Connect to single logit
        with tf.variable_scope('output', reuse=reuse):
            output = tf.layers.dense(output, 1)[:, 0]

        # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

        return output


"""
  Input: [None, 100]
  Output: [None, 16384, 1]
"""
def generator_wavegan(
    z,
    kernel_len=25,
    dim=64,
    use_batchnorm=True,
    upsample='zeros',
    train=False,
    scope='Generator'
    ):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(z)[0]

        if use_batchnorm:
            batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
        else:
            batchnorm = lambda x: x

        # FC and reshape for convolution
        # [100] -> [16, 1024]
        output = z
        with tf.variable_scope('z_project'):
            output = tf.layers.dense(output, 4 * 4 * dim * 16)
            output = tf.reshape(output, [batch_size, 16, dim * 16])
            output = batchnorm(output)
        output = tf.nn.relu(output)

        # Layer 0
        # [16, 1024] -> [64, 512]
        with tf.variable_scope('upconv_0'):
            output = conv1d_transpose(output, dim * 8, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)

        # Layer 1
        # [64, 512] -> [256, 256]
        with tf.variable_scope('upconv_1'):
            output = conv1d_transpose(output, dim * 4, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)

        # Layer 2
        # [256, 256] -> [1024, 128]
        with tf.variable_scope('upconv_2'):
            output = conv1d_transpose(output, dim * 2, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)

        # Layer 3
        # [1024, 128] -> [4096, 64]
        with tf.variable_scope('upconv_3'):
            output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
            output = batchnorm(output)
        output = tf.nn.relu(output)

        # Layer 4
        # [4096, 64] -> [16384, 1]
        with tf.variable_scope('upconv_4'):
            output = conv1d_transpose(output, 1, kernel_len, 4, upsample=upsample)
        output = tf.nn.tanh(output)

        # Automatically update batchnorm moving averages every time G is used during training
        if train and use_batchnorm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if len(update_ops) != 10:
                raise Exception('Other update ops found in graph')
            with tf.control_dependencies(update_ops):
                output = tf.identity(output)

        return output


# def generator(x, is_training=True, scope='Generator'):
#   # fc1024-bn-relu + fc6272-bn-relu + deconv64-bn-relu + deconv1-tanh
#   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#     x = _dense(x, 1024, name='g_fc1')
#     x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))
#
#     x = _dense(x, 7 * 7 * 128, name='g_fc2')
#     x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))
#
#     x = tf.reshape(x, [-1, 7, 7, 128])
#
#     x = _deconv2d(x, 64, 4, 2, name='g_dconv3')
#     x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))
#
#     x = _deconv2d(x, 1, 4, 2, name='g_dconv4')
#     x = tf.tanh(x)
#
#     return x

# TODO(chrisying): objective score (e.g. MNIST score)

