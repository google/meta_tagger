# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def linear_with_dropout(is_training,
                        inputs,
                        output_size,
                        initializer=None,
                        keep_prob=1,
                        add_bias=True):
  """Linear mapping with dropout."""
  input_size = inputs.get_shape().as_list()[-1]

  if is_training and keep_prob < 1:
    inputs = tf.nn.dropout(inputs, keep_prob)

  shape = tf.shape(inputs)
  output_shape = []
  for i in xrange(len(inputs.get_shape().as_list()) - 1):
    output_shape.append(shape[i])
  output_shape.append(output_size)

  inputs = tf.reshape(inputs, [-1, input_size])
  if not initializer:
    initializer = tf.orthogonal_initializer()

  with tf.variable_scope('Linear'):
    matrix = tf.get_variable(
        'Weights', [input_size, output_size], initializer=initializer)
    # Get the bias
    if add_bias:
      bias = tf.get_variable(
          'Biases', [output_size], initializer=tf.zeros_initializer())
    else:
      bias = 0

    # Do the multiplication
    linear = tf.nn.xw_plus_b(inputs, matrix, bias)
    return tf.reshape(linear, output_shape)


def lstm_layers(is_training,
                inputs,
                num_layers,
                hidden_size,
                recur_keep_prob,
                sequence_length=None):
  """Defines the LSTM layers."""
  outputs = inputs
  for n in range(num_layers):
    cudnn_lstms = True
    if cudnn_lstms:
      cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
      cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
    else:
      cell_fw = tf.contrib.rnn.LSTMCell(hidden_size, use_peepholes=True)
      cell_bw = tf.contrib.rnn.LSTMCell(hidden_size, use_peepholes=True)

    if is_training and recur_keep_prob < 1:
      input_size = outputs.get_shape().as_list()[-1]
      cell_fw = tf.contrib.rnn.DropoutWrapper(
          cell_fw,
          input_keep_prob=recur_keep_prob,
          state_keep_prob=recur_keep_prob,
          variational_recurrent=True,
          dtype=tf.float32,
          input_size=input_size)
      cell_bw = tf.contrib.rnn.DropoutWrapper(
          cell_bw,
          input_keep_prob=recur_keep_prob,
          state_keep_prob=recur_keep_prob,
          variational_recurrent=True,
          dtype=tf.float32,
          input_size=input_size)

    batch_size = tf.shape(outputs)[0]
    (output_fw, output_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        outputs,
        sequence_length=sequence_length,
        initial_state_fw=cell_fw.zero_state(batch_size, tf.float32),
        initial_state_bw=cell_bw.zero_state(batch_size, tf.float32),
        time_major=False,
        scope='bilstm_' + str(n),
        dtype=tf.float32)
    outputs = tf.concat([output_fw, output_bw], axis=2)
  return output_fw, output_bw, final_state


def mlp(is_training,
        inputs,
        output_size=None,
        keep_prob=None,
        add_bias=True):
  """Multi layer perceptron."""
  with tf.variable_scope('MLP'):
    linear = linear_with_dropout(
        is_training,
        inputs,
        output_size,
        keep_prob=keep_prob,
        add_bias=add_bias)
    return tf.nn.elu(linear)


def random_mask(prob, mask_shape, dtype=tf.float32):
  """Random mask."""

  rand = tf.random_uniform(mask_shape)
  ones = tf.ones(mask_shape, dtype=dtype)
  zeros = tf.zeros(mask_shape, dtype=dtype)
  prob = tf.ones(mask_shape) * prob
  return tf.where(rand < prob, ones, zeros)


def dropout(is_training, embed_keep_prob, inputs):
  if is_training and embed_keep_prob < 1:
    if len(inputs.get_shape().as_list()) == 3:
      ph = tf.unstack(inputs, axis=2)[0]
      #ph = tf.shape(inputs)[-1]
      drop_mask = tf.expand_dims(random_mask(embed_keep_prob, tf.shape(ph)), 2)
      inputs *= drop_mask
  return inputs
