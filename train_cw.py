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

from datetime import timedelta
import json
import math
import os.path
import timeit
from absl import flags
import layers
import numpy as np
import reader as rd
import tensorflow as tf
import test as tester

logging = tf.logging

flags.DEFINE_string(
    'train',
    '~/corpora/data/PTB/en-wsj-std-train-stanford-3.3.0.conll',
    'Filename of training data')

flags.DEFINE_string(
    'test',
    '~/`en-wsj-std-test-stanford-3.3.0.conll',
    'Filename of testing data (optional)')

flags.DEFINE_string(
    'dev',
    '~/corpora/data/PTB/en-wsj-std-dev-stanford-3.3.0.conll',
    'Filename of development data')
flags.DEFINE_string(
    'embeddings',
    '~/corpora/data/glove/glove.6B.100d.txt', 'Path to embeddings')
flags.DEFINE_string('output_dir', '~/tensorboard/wsj_xtag_v2/', '')

flags.DEFINE_string('task', 'xtag', 'xtag|tag|mor')

flags.DEFINE_boolean('testing', False, '')

flags.DEFINE_string('config', 'config.json', 'Jason configuration file name.')

FLAGS = flags.FLAGS

PAD = 0
UNK = 1


def parameters():
  hparams = tf.contrib.training.HParams(
      learning_rate=2e-3,
      hidden_char_size=300,
      hidden_word_size=500,
      hidden_meta_size=500,
      mlp_size=300,
      keep_prob=0.67,
      embed_keep_prob=0.67,
      embed_keep_prob_ch=0.95,
      recur_keep_prob=0.67,
      recur_keep_j_prob=0.67,
      recur_keep_w_prob=0.67,
      num_layers_chars=3,
      num_layers_words=2,
      num_layers_meta=1,
      learning_rate_decay=0.999994,
      batch_word_size=5000,
      batch_char_size=10000,
      embed_size=100,
      min_occurrence=2,
      lowercase=True,
      tagging=4,
      early_stopping_steps=0,
      task_name='meta_word_char_v1')
  if tf.gfile.Exists(FLAGS.config):
    params_json = ''
    for line in rd.read_file_to_stringio(FLAGS.config):
      params_json += line
    hparams.parse_json(params_json)
  return hparams


def log_training_time(start, num_batches, batch_size):
  time_train = (timeit.default_timer() - start)
  time_token = (num_batches * batch_size) / time_train
  logging.info('training: %0.1f sec %f tok/sec' % (time_train, time_token))


class Accuracies(object):
  """Holds evaluation accuracies scores."""

  def __init__(self):
    self.meta = 0.0
    self.word = 0.0
    self.char = 0.0


class Vocab(object):
  """Holds vocabolary."""

  def __init__(self):
    self.word_id = {}
    self.char_id = {}
    self.pred_id = {}
    self.tag_id = {}
    self.id_tag = {}

  def write(self, output_dir):
    output_dir = os.path.expanduser(output_dir)
    with tf.gfile.GFile(os.path.join(output_dir, 'word_id.txt'), 'w') as f:
      f.write(json.dumps(self.word_id))
    with tf.gfile.GFile(os.path.join(output_dir, 'char_id.txt'), 'w') as f:
      f.write(json.dumps(self.char_id))
    with tf.gfile.GFile(os.path.join(output_dir, 'tag_id.txt'), 'w') as f:
      f.write(json.dumps(self.tag_id))
    with tf.gfile.GFile(os.path.join(output_dir, 'pred_id.txt'), 'w') as f:
      f.write(json.dumps(self.pred_id))

  def read(self, output_dir):
    def read_dict(dictonary, filename):
      output_json = json.load(tf.gfile.GFile(output_dir + filename, 'r'))
      for key, val in output_json.iteritems():
        dictonary[key] = val
    read_dict(self.tag_id, 'tag_id.txt')
    self.id_tag = dict(map(reversed, self.tag_id.items()))
    read_dict(self.word_id, 'word_id.txt')
    read_dict(self.char_id, 'char_id.txt')
    read_dict(self.pred_id, 'pred_id.txt')

  def init_id_tag(self):
    self.id_tag = dict(map(reversed, self.tag_id.items()))


class Model(object):
  """Model for training and testing."""

  def __init__(self, hp):
    self.inputs_words = tf.placeholder(
        dtype=tf.int32, shape=(None, None, None), name='inputs_words')
    self.inputs_chars = tf.placeholder(
        dtype=tf.int32, shape=(None, None), name='inputs_chars')
    self.targets_ch = tf.placeholder(
        dtype=tf.int32, shape=(None, None), name='targets_chars')
    self.idx_start = tf.placeholder(
        dtype=tf.int32, shape=(None, None, 2), name='indexs_start')
    self.idx_end = tf.placeholder(
        dtype=tf.int32, shape=(None, None, 2), name='indexs_end')
    self.lout_w = tf.placeholder(
        dtype=tf.float32, shape=(None, None, hp.mlp_size), name='loutw')
    self.lout_c = tf.placeholder(
        dtype=tf.float32, shape=(None, None, hp.mlp_size), name='loutc')
    self.acc_m = tf.placeholder(dtype=tf.float32, shape=(), name='acc-meta-dev')
    self.acc_c = tf.placeholder(dtype=tf.float32, shape=(), name='acc-char-dev')
    self.acc_w = tf.placeholder(dtype=tf.float32, shape=(), name='acc-word-dev')
    self.acc_its = tf.placeholder(dtype=tf.float32, shape=(), name='iterations')

  def summaries_acc(self, loss, name='name'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries/'):
      tf.summary.scalar(name, loss)
    return

  def create_summary(self, merged, sess, ph, acc):
    return sess.run(
        merged,
        feed_dict={ph.acc_m: acc.meta, ph.acc_c: acc.char, ph.acc_w: acc.word})

  def char_model(self, is_training, hparams, chars, embedding_char_size, tags,
                 inputs_char, indexs_start, indexs_end, targets_w):
    """Character model."""
    with tf.variable_scope('chars'):
      if is_training:
        embed_dims = [chars, embedding_char_size]
        np.random.seed(seed=1)
        embeddings_char = np.random.randn(*embed_dims).astype(np.float32)
        cembed = tf.get_variable(
            'char_embeddings', dtype=tf.float32, initializer=embeddings_char)
      else:
        cembed = tf.get_variable('char_embeddings')

      # joint for both
      embed_nd = tf.nn.embedding_lookup(cembed, inputs_char[:, :])
      embed = layers.dropout(is_training, hparams.embed_keep_prob_ch, embed_nd)

      output_fw, output_bw, _ = layers.lstm_layers(
          is_training, embed, hparams.num_layers_chars,
          hparams.hidden_char_size, hparams.recur_keep_prob)

      # Gather forward start and end of word of char LSTM output.
      output_fw_fst = tf.gather_nd(output_fw, indexs_start)
      output_fw_lst = tf.gather_nd(output_fw, indexs_end)

      # Gather backword start and end of word of char LSTM output.
      output_bw_fst = tf.gather_nd(output_bw, indexs_start)
      output_bw_lst = tf.gather_nd(output_bw, indexs_end)

      # Gathered LSTM outputs into the right shape and concatenate it.
      outputs = tf.concat(
          [output_fw_fst, output_fw_lst, output_bw_fst, output_bw_lst], axis=2)

      outputs = layers.mlp(
          is_training,
          outputs,
          output_size=hparams.mlp_size,
          keep_prob=hparams.keep_prob)

      targets = targets_w[:, :]
      tok_keep = tf.to_float(tf.greater(targets, PAD))

      linear = layers.linear_with_dropout(
          is_training, outputs, tags, keep_prob=hparams.keep_prob)
      preds = tf.to_int32(tf.argmax(linear, axis=-1))

      if is_training:
        int_tok_keep = tf.to_int32(tok_keep)
        t_correct = tf.to_int32(tf.equal(preds, targets)) * int_tok_keep
        accuracy = tf.reduce_sum(t_correct) / tf.reduce_sum(int_tok_keep)

        loss = tf.losses.sparse_softmax_cross_entropy(targets, linear, tok_keep)
        return loss, accuracy
      else:
        return preds, outputs

  def word_model(self, is_training, hparams, words, embedding_word_size,
                 tags, pretrained_embed, inputs):
    """Word model."""
    with tf.variable_scope('words'):
      embedding = tf.get_variable(
          'word_embedding', [words, embedding_word_size],
          dtype=tf.float32,
          initializer=tf.zeros_initializer())
      word_inputs = tf.nn.embedding_lookup(embedding, inputs[:, :, 0])
      word_inputs = word_inputs
      word_inputs = layers.dropout(is_training, hparams.embed_keep_prob,
                                   word_inputs)
      pret_inputs = tf.nn.embedding_lookup(pretrained_embed, inputs[:, :, 1])
      pret_inputs = layers.dropout(is_training, hparams.embed_keep_prob,
                                   pret_inputs)
      word_inputs += pret_inputs

      targets_w = inputs[:, :, 2]
      outputs = word_inputs

      output_fw, output_bw, _ = layers.lstm_layers(
          is_training, outputs, hparams.num_layers_words,
          hparams.hidden_word_size, hparams.recur_keep_w_prob)
      outputs = tf.concat([output_fw, output_bw], axis=2)

      outputs = layers.mlp(
          is_training,
          outputs,
          output_size=hparams.mlp_size,
          keep_prob=hparams.keep_prob)

      logits = layers.linear_with_dropout(
          is_training,
          outputs,
          tags,
          keep_prob=hparams.keep_prob)
      preds_w = tf.to_int32(tf.argmax(logits, axis=-1))
      tag_correct_w = tf.to_int32(tf.equal(preds_w, targets_w))
      correct = tf.reduce_sum(tag_correct_w) / tf.size(
          tag_correct_w)
      tokens_to_keep = tf.to_float(tf.greater(inputs[:, :, 0], PAD))
      loss_w = tf.losses.sparse_softmax_cross_entropy(targets_w, logits,
                                                      tokens_to_keep)

      if is_training:
        return loss_w, correct
      else:
        return preds_w, outputs

  def join(self, is_training, hparams, inputs, out_w, out_c, tags):
    """Meta model joins word and char model."""
    with tf.variable_scope('meta_char_word'):
      out_1 = layers.dropout(is_training, hparams.keep_prob, out_w)
      out_2 = layers.dropout(is_training, hparams.keep_prob, out_c)

      outputs = tf.concat([out_1, out_2], axis=2)
      out_fw, out_bw, _ = layers.lstm_layers(is_training, outputs,
                                             hparams.num_layers_meta,
                                             hparams.hidden_meta_size,
                                             hparams.recur_keep_j_prob)
      outputs = tf.concat([out_fw, out_bw], axis=2)
      outputs = layers.mlp(
          is_training,
          outputs,
          output_size=tags,
          keep_prob=hparams.keep_prob)
      preds_w = tf.to_int32(tf.argmax(outputs, axis=-1))
      targets_w = inputs[:, :, 2]
      tokens_to_keep = tf.to_float(tf.greater(inputs[:, :, 0], PAD))
      loss = tf.losses.sparse_softmax_cross_entropy(targets_w, outputs,
                                                    tokens_to_keep)
    if is_training:
      return loss
    else:
      return preds_w


def test(sess, evaluate, ph, dataset, testmodel):
  """Apply the models."""
  ## word model
  acc = Accuracies()
  out_sentences = []
  results_w = []
  for f_word in dataset.batches:
    batch_values, out_logits_w_out_w = sess.run(
        [testmodel.predictions_w, testmodel.out_logits_w],
        feed_dict={ph.inputs_words: f_word})
    results_w.extend(out_logits_w_out_w)

    for a in batch_values:
      out_sentences.append([w for w in a])
  acc.word = evaluate.simple_eval(out_sentences)

  ## char model
  out_sentences = []
  results_c = []
  for f_char, i_start, i_end in zip(
      dataset.batches_ch, dataset.index_batches_start,
      dataset.index_batches_end):
    feed = {ph.inputs_chars: f_char, ph.idx_start: i_start, ph.idx_end:
            i_end}
    batch_values, ch_out = sess.run(
        [testmodel.predictions_c, testmodel.out_logits_c],
        feed_dict=feed)
    results_c.extend(ch_out)

    for a in batch_values:
      out_sentences.append([w for w in a])
  acc.char = evaluate.simple_eval(out_sentences)

  ## join models
  out_sentences = []
  index_step = 0
  for batch_w in dataset.batches:
    cout, wout = ([], [])
    for _ in batch_w:
      wout.append(results_w[index_step])
      lsc = results_c[index_step]
      w_shape = results_w[index_step].shape
      pad_c = np.zeros(w_shape)
      if w_shape[0] <= lsc.shape[0]:
        pad_c[:w_shape[0], :w_shape[1]] = lsc[:w_shape[0], :w_shape[1]]
      else:
        pad_c[:lsc.shape[0], :lsc.shape[1]] = lsc
      cout.append(pad_c)
      index_step += 1
    feed = {ph.inputs_words: batch_w, ph.lout_w: wout, ph.lout_c: cout}
    batch_values_joint = sess.run(testmodel.predictions_m, feed_dict=feed)

    for a in batch_values_joint:
      out_sentences.append([w for w in a])
  test_stringio_joint = evaluate.write_string(out_sentences)
  acc.meta = evaluate.simple_eval(out_sentences)
  return acc, test_stringio_joint


class Dataset(object):

  def __init__(self, data, vocab, reader, sp_char, hparams):
    sentences_id = reader.sentences_ids(data, vocab.word_id, vocab.tag_id)
    char_data = reader.to_char_corpus(data, tag_position=1)
    self.batches = reader.sentences_to_buckets(hparams, sentences_id)
    sentences_char = reader.char_sentences(char_data, vocab.char_id,
                                           vocab.tag_id)
    (self.batches_ch, self.index_batches_end, self.index_batches_start,
     self.targets_chars) = (
         reader.char_sentences_to_buckets_index_sc(hparams, sentences_char,
                                                   sp_char))


class SntId(object):
  """Stores sentences with and index for restoring the order."""

  def __init__(self, snt, index):
    self.index = index
    self.snt = snt


def run_training():
  """Main method for training and testing."""
  logging.set_verbosity(logging.INFO)

  # Get configuration and read additional parameters for json file.
  hparams = parameters()
  keys = hparams.values().keys()
  keys.sort()
  tw_name = hparams.task_name
  for key in keys:
    value = hparams.values()[key]
    logging.info(' '+ key + ('\t' if len(key) > 14 else '\t\t') + str(value))
    if key.startswith('batch'):
      continue
    if isinstance(value, int) or isinstance(value, float):
      tw_name += '_'
      for s in key.split('_'):
        tw_name += s[0]
      tw_name += '%d' % value if isinstance(value, int) else '%.3f' % value
  logging.info('tensorboard name %s' % tw_name)

  task_dict = {'upos': 3, 'xtag': 4, 'feats': 5}
  hparams.tagging = task_dict[FLAGS.task]
  conll_columns = [1, hparams.tagging]

  # Uses min_occurrence, lowercase, batch_char_size, batch_word_size
  reader = rd.Reader(hparams)

  logging.info('reading embeddings %s' % FLAGS.embeddings)
  reader.load(FLAGS.embeddings)

  logging.info('reading training data %s' % FLAGS.train)
  data = reader.read_corpus(FLAGS.train, conll_columns)
  sentences_train = reader.sentences(data)

  ## Create vocab and batch training data. ##
  vocab = Vocab()
  vocab.word_id = reader.build_word_vocab(sentences_train, hparams.lowercase)
  char_train = reader.to_char_corpus(data, tag_position=1)
  vocab.char_id, vocab.tag_id = reader.build_char_vocab(char_train,
                                                        add_special_tokens=True)
  vocab.init_id_tag()
  output_dir = os.path.expanduser(FLAGS.output_dir)
  vocab.pred_id = reader.embedding_dict
  vocab.write(FLAGS.output_dir)

  # list of sentences
  sentences_char_train = reader.char_sentences(char_train, vocab.char_id,
                                               vocab.tag_id)
  sentences_char_train_index = []
  for k, s in enumerate(sentences_char_train):
    sentences_char_train_index.append(SntId(s, k))
  sort_len = lambda snt_id: len(snt_id.snt)
  sentences_char_train_index.sort(key=sort_len)

  # Index from batch-index to index in training set.
  sentences_char_index_b2t = {}
  sentences_char_index_t2b = {}
  sentences_char_train = []
  for k, s in enumerate(sentences_char_train_index):
    sentences_char_index_b2t[k] = s.index
    sentences_char_index_t2b[s.index] = k
    sentences_char_train.append(s.snt)

  # Create the char batches.
  batches_char, batches_idx_end, batches_idx_start, target_batch = (
      reader.char_sentences_to_buckets_index_sc(hparams, sentences_char_train,
                                                vocab.char_id['\t']))

  # Collect the sentences and sort due to length.
  sentences_id = reader.sentences_ids(data, vocab.word_id, vocab.tag_id)

  # Evaluate number of unknown words.
  reader.check_for_unknown_pretrained_embeddings(sentences_id, vocab.word_id)

  sentences_id_index = [SntId(s, k) for k, s in enumerate(sentences_id)]
  sentences_id_index.sort(key=sort_len)
  dev = tester.Test(hparams, reader, FLAGS.dev, vocab.id_tag)

  # Maps sentence to the batches.
  sentence_index = {}
  sentences_id = []
  for k, si in enumerate(sentences_id_index):
    sentences_id.append(si.snt)
    sentence_index[k] = si.index
  sentences_id.sort(key=len)
  batches_word = reader.sentences_to_buckets(hparams, sentences_id)

  # Read development data.
  dev_data = reader.read_corpus(FLAGS.dev, conll_columns)
  dev_dataset = Dataset(dev_data, vocab, reader, vocab.char_id['\t'], hparams)

  # Builds model for training.
  global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
  global_step_tensor_joint = tf.Variable(
      0, trainable=False, name='global_step_joint')
  global_step_tensor_ch = tf.Variable(0, trainable=False, name='global_step_ch')

  # Adam has his own adaptive learning rate, nevertheless use a bit decay.
  steps_decay = math.ceil(float(hparams.batch_word_size) / 5000.0)
  logging.info('steps_decay %d ' % steps_decay)
  rate = tf.train.exponential_decay(hparams.learning_rate,
                                    global_step_tensor, steps_decay,
                                    hparams.learning_rate_decay)
  optimizer = tf.contrib.opt.LazyAdamOptimizer(rate)

  # Optimizer for char model
  steps_decay_ch = math.ceil(float(hparams.batch_char_size) / 5000.0)
  rate_ch = tf.train.exponential_decay(
      hparams.learning_rate, global_step_tensor_ch, steps_decay_ch,
      hparams.learning_rate_decay)
  optimizer_ch = tf.contrib.opt.LazyAdamOptimizer(rate_ch)

  # Optimizer for word model
  rate_joint = tf.train.exponential_decay(
      hparams.learning_rate, global_step_tensor_joint, steps_decay,
      hparams.learning_rate_decay)

  # Optimizer for meta model
  optimizer_joint = tf.contrib.opt.LazyAdamOptimizer(rate_joint)
  model = Model(hparams)

  logging.info('Create training model.')
  pretrained = tf.Variable(
      tf.constant(0.0, shape=[len(reader.embeddings), reader.embed_size]),
      trainable=False,
      name='Pre')
  ph = model
  with tf.variable_scope('x', reuse=None):
    is_training = True
    (loss_w,
     correct) = model.word_model(is_training, hparams,
                                 len(vocab.word_id), reader.embed_size,
                                 len(vocab.tag_id), pretrained, ph.inputs_words)

    (loss_ch, correct_ch) = model.char_model(
        is_training, hparams, len(vocab.char_id), hparams.embed_size,
        len(vocab.tag_id), ph.inputs_chars, ph.idx_start, ph.idx_end,
        ph.targets_ch)

    loss_m = model.join(is_training, hparams, ph.inputs_words, ph.lout_w,
                        ph.lout_c, len(vocab.tag_id))

  # Setup the optimizers.
  train = optimizer.minimize(loss_w, global_step=global_step_tensor)
  traim_m = optimizer_joint.minimize(loss_m,
                                     global_step=global_step_tensor_joint)
  train_ch = optimizer_ch.minimize(loss_ch, global_step=global_step_tensor_ch)

  # Prepare the embeddings.
  embedding_ph = tf.placeholder(tf.float32, [len(reader.embeddings),
                                             reader.embed_size])
  embedding_init = pretrained.assign(embedding_ph)

  logging.info('Create testing model.')
  with tf.variable_scope('x', reuse=True):
    is_training = False
    (predictions_w,
     out_logits_w) = model.word_model(
         is_training, hparams, len(vocab.word_id), reader.embed_size,
         len(vocab.tag_id), pretrained, ph.inputs_words)

    (predictions_c, out_logits_c) = model.char_model(
        is_training, hparams, len(vocab.char_id), hparams.embed_size,
        len(vocab.tag_id), ph.inputs_chars, ph.idx_start, ph.idx_end,
        ph.targets_ch)

    predictions_m = model.join(is_training, hparams, ph.inputs_words, ph.lout_w,
                               ph.lout_c, len(vocab.tag_id))

  class Testmodel(object):
    """Tensors for testmodel"""

    def __init__(self):
      self.predictions_w = predictions_w
      self.out_logits_w = out_logits_w
      self.predictions_c = predictions_c
      self.out_logits_c = out_logits_c
      self.predictions_m = predictions_m
  testmodel = Testmodel()

  model.summaries_acc(ph.acc_m, 'dev-' + FLAGS.task)
  model.summaries_acc(ph.acc_c, 'dev-ch-' + FLAGS.task)
  model.summaries_acc(ph.acc_w, 'dev-w-' + FLAGS.task)

  saver = tf.train.Saver()

  # Create session and initialize variables.
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())

    # Load embeddings into TF.
    sess.run(embedding_init, feed_dict={embedding_ph: reader.embeddings})
    iteration = 0
    start = timeit.default_timer()

    # Merge all the summaries.
    merged = tf.summary.merge_all()

    logging.info('Writing summary to ' + os.path.join(output_dir, tw_name))
    train_writer = tf.summary.FileWriter(os.path.join(output_dir, tw_name),
                                         sess.graph)

    filename_model = os.path.join(output_dir, str(hparams.task_name))

    # Continue training if model exists.
    logging.info('check existens %s' % (filename_model + '.meta'))

    if tf.gfile.Exists(filename_model + '.meta'):
      logging.info('found existing model')
      saver.restore(sess, os.path.join(output_dir, str(hparams.task_name)))

      best_acc, _ = test(sess, dev, ph, dev_dataset, testmodel)
      logging.info('acc_dev %f'  % best_acc.meta)

    else:
      best_acc = Accuracies()
    state = 'train_word_model'
    max_steps_to_try_improve = hparams.early_stopping_steps
    word_improve_steps = max_steps_to_try_improve + 500
    char_improve_steps = max_steps_to_try_improve + 500
    joint_improve_steps = max_steps_to_try_improve + 500

    start = timeit.default_timer()
    batches_results_c, batches_results_w = ([], [])

    total_time = timeit.default_timer()

    # Training loop.
    while True:
      iteration += 1
      logging.info('Iteration %d ' % iteration)

      start = timeit.default_timer()
      num_batches = 0
      logging.info(
          'total time %s' %
          str(timedelta(seconds=(timeit.default_timer() - total_time))))

      # Train word model.
      if state == 'train_word_model':
        for fw in batches_word:
          _, loss_out, correct_out = sess.run(
              [train, loss_w, correct], feed_dict={ph.inputs_words: fw})
          num_batches += 1

          steps = sess.run(global_step_tensor)
          word_improve_steps -= 1
          if steps % 5 == 0:
            logging.info('steps word: %d loss %f correct %f ' %
                         (steps, loss_out, correct_out))
        log_training_time(start, num_batches, hparams.batch_word_size)

        acc, _ = test(sess, dev, ph, dev_dataset, testmodel)
        if best_acc.word < acc.word:
          best_acc.word = acc.word
          if word_improve_steps < max_steps_to_try_improve:
            word_improve_steps = max_steps_to_try_improve
          saver.save(sess, filename_model)
          logging.info('model saved')

        logging.info('dev set: word model %f' % acc.word)

        summary = model.create_summary(merged, sess, ph, acc)
        train_writer.add_summary(summary, iteration)
        logging.info('best word accuracy %f and steps to go %d', best_acc.word,
                     word_improve_steps)
        if word_improve_steps <= 0:
          state = 'train_char_model'
          logging.info('load best model.')
          saver.restore(sess, filename_model)
        else:
          continue

      # train the char model
      if state == 'train_char_model':
        for fc, i_end, i_start, target in zip(batches_char, batches_idx_end,
                                              batches_idx_start, target_batch):
          feed = {ph.inputs_chars: fc, ph.idx_start: i_start, ph.idx_end: i_end,
                  ph.targets_ch: target}
          sess_out = sess.run(
              [train_ch, loss_ch, correct_ch], feed_dict=feed)
          num_batches += 1

          char_improve_steps -= 1
          steps_ch = sess.run(global_step_tensor_ch)
          if steps_ch % 10 == 0:
            logging.info('steps chars: %d loss %f correct %f ' %
                         (steps_ch, sess_out[1], sess_out[2]))

        acc, _ = test(sess, dev, ph, dev_dataset, testmodel)
        log_training_time(start, num_batches, hparams.batch_char_size)

        if best_acc.char < acc.char:
          best_acc.char = acc.char
          if char_improve_steps < max_steps_to_try_improve:
            char_improve_steps = max_steps_to_try_improve
          saver.save(sess, filename_model)
          logging.info('model saved')

        logging.info('dev accuracies: meta %f word %f char %f' %
                     (acc.meta, acc.word, acc.char))

        summary = model.create_summary(merged, sess, ph, acc)
        train_writer.add_summary(summary, iteration)

        logging.info('best char accuracy %f and steps to go %d', best_acc.char,
                     char_improve_steps)
        if char_improve_steps <= 0:
          state = 'train_joint_model'
          logging.info('load best char model and continue with meta model')
          start = timeit.default_timer()
          saver.restore(sess, filename_model)
        else:
          continue

      # Rerun the models with the test model for training the meta model.
      if not batches_results_c:
        results_w, results_ch = ([], [])
        logging.info('compute word model')
        for fw in batches_word:
          feed = {ph.inputs_words: fw}
          results_w.extend(sess.run(out_logits_w, feed_dict=feed))

        logging.info('compute char model')
        for fc, e_idx, s_idx in zip(batches_char, batches_idx_end,
                                    batches_idx_start):
          feed = {ph.inputs_chars: fc, ph.idx_start: s_idx, ph.idx_end: e_idx}
          results_ch.extend(sess.run(out_logits_c, feed_dict=feed))

        index_step = 0
        # Since sentences of word and char batches are sorted differently,
        # merge them in order to run the meta model.
        for batch_w in batches_word:
          cout, wout = ([], [])
          for _ in batch_w:
            wout.append(results_w[index_step])
            w_train_index = sentence_index[index_step]
            lsc = results_ch[sentences_char_index_t2b[w_train_index]]
            w_shape = results_w[index_step].shape
            pad_c = np.zeros(w_shape)
            if w_shape[0] <= lsc.shape[0]:
              pad_c[:w_shape[0], :w_shape[1]] = lsc[:w_shape[0], :w_shape[1]]
            else:
              pad_c[:lsc.shape[0], :lsc.shape[1]] = lsc
            cout.append(pad_c)
            index_step += 1
          batches_results_c.append(cout)
          batches_results_w.append(wout)

      # Train the meta model.
      for batch_w, cout, wout in zip(batches_word, batches_results_c,
                                     batches_results_w):
        feed = {ph.inputs_words: batch_w, ph.lout_w: wout, ph.lout_c: cout}
        _, loss_joint = sess.run([traim_m, loss_m], feed_dict=feed)
        num_batches += 1
        joint_improve_steps -= 1
        total_train_iters_joint = sess.run(global_step_tensor_joint)
        if total_train_iters_joint % 5 == 0:
          logging.info('steps meta: %d loss %f ',
                       total_train_iters_joint, loss_joint)

      log_training_time(start, num_batches, hparams.batch_word_size)
      acc, _ = test(sess, dev, ph, dev_dataset, testmodel)

      if best_acc.meta < acc.meta:
        if joint_improve_steps < max_steps_to_try_improve:
          joint_improve_steps = max_steps_to_try_improve
        best_acc = acc
        saver.save(sess, filename_model)
        logging.info('model saved')
      logging.info('dev accuracies: meta %f word %f char %f' %
                   (acc.meta, acc.word, acc.char))
      logging.info('best meta model accuracy %f and steps to go %d',
                   best_acc.meta, joint_improve_steps)
      summary = model.create_summary(merged, sess, ph, acc)
      train_writer.add_summary(summary, iteration)
      if joint_improve_steps <= 0:
        logging.info(
            'total time %s' %
            str(timedelta(seconds=(timeit.default_timer() - total_time))))
        return


def main(argv):
  """Run training or testing."""
  del argv  # Unused.
  run_training()

if __name__ == '__main__':
  tf.app.run()
