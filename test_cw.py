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

import codecs
import os
from absl import flags
import numpy as np
import reader as rd
import tensorflow as tf

import train_cw
import test as tester

flags.DEFINE_string('out', '', 'Name of the output file.')

FLAGS = flags.FLAGS


def read_corpus(filename, char_id, word_id, pred_id, idx=1, space=u'\t'):
  """Reads a corpus in CoNLL format."""
  filename = os.path.expanduser(filename)
  data, data_c, snt, sntc, start, end = ([], [], [], [], [], [])
  def add_dataset_and_index(snt, sntc):
    sstart, send = ([[0, 0]], [])
    for i, c in enumerate(sntc, 0):
      if c == char_id[u'\t']:
        send.append([0, i-1])
        if i != len(sntc) -1:
          sstart.append([0, i+1])
    start.append([sstart])
    end.append([send])
    data.append([snt])
    data_c.append([sntc])
    return ([], [])

  for line in codecs.getreader('utf-8')(tf.gfile.GFile(filename, 'r')):
    if line.strip().startswith(u'#'): continue
    if len(line.split(u'\t')) > 4:
      snt.append([word_id.get(line.split(u'\t')[idx].lower(), 0),
                  pred_id.get(line.split(u'\t')[idx].lower(), 0)])
      sntc.extend([char_id.get(c, 0) for c in line.split(u'\t')[idx] + space])
    else:
      snt, sntc = add_dataset_and_index(snt, sntc)
  if snt:
    snt, sntc = add_dataset_and_index(snt, sntc)
  return (data, data_c, start, end)


class Vocab(object):
  """Loading and storing vocabulary."""

  def __init__(self, output_dir):
    (self.word_id, self.char_id, self.pred_id, self.tag_id,
     self.id_tag) = ({}, {}, {}, {}, {})
    def read(dictonary, filename):
      output_json = json.load(tf.gfile.GFile(output_dir + filename, 'r'))
      for key, val in output_json.iteritems():
        dictonary[key] = val
    read(self.tag_id, 'tag_id.txt')
    self.id_tag = dict(map(reversed, self.tag_id.items()))
    read(self.word_id, 'word_id.txt')
    read(self.char_id, 'char_id.txt')
    read(self.pred_id, 'pred_id.txt')


def run_testing():
  """Execute testing."""
  hparams = train_cw.parameters()
  task_dict = {'upos': 3, 'xtag': 4, 'feats': 5}
  hparams.tagging = task_dict[FLAGS.task]
  conll_columns = [1, hparams.tagging]

  output_dir = os.path.expanduser(FLAGS.output_dir)
  voc = train_cw.Vocab()
  voc.read(output_dir)
  reader = rd.Reader(hparams)

  test_set = reader.read_corpus(FLAGS.test, conll_columns)
  char_test = reader.to_char_corpus(test_set, tag_position=1)
  sentences_char_test = reader.char_sentences(char_test, voc.char_id,
                                              voc.tag_id)
  batches_c, batches_end, batches_start, _ = (
      reader.char_sentences_to_buckets_index_sc(hparams, sentences_char_test,
                                                voc.char_id[u'\t']))
  test_sentences_id = reader.sentences_ids(test_set, voc.word_id, voc.tag_id,
                                           voc.pred_id)
  batches_w = reader.sentences_to_buckets(hparams, test_sentences_id)

  tes = tester.Test(hparams, reader, FLAGS.test, voc.id_tag)
  with tf.Session() as s:

    filename_model = os.path.join(output_dir, str(hparams.task_name))
    saver = tf.train.import_meta_graph(filename_model + '.meta')
    s.run(tf.global_variables_initializer())
    saver.restore(s, filename_model)
    ph_names = ['inputs_words', 'inputs_chars', 'indexs_start', 'indexs_end',
                'loutw', 'loutc']
    graph = tf.get_default_graph()
    ph = {name: graph.get_tensor_by_name(name + ':0') for name in ph_names}
    results_w = []

    out_sentences = []

    for batch_w in batches_w:
      results_w.extend(s.run(graph.get_tensor_by_name('x_1/words/MLP/Elu:0'),
                             feed_dict={ph['inputs_words']: batch_w}))
    results_c = []
    for batch_c, start, end in zip(batches_c, batches_start, batches_end):
      feed = {ph['inputs_chars']: batch_c, ph['indexs_start']: start,
              ph['indexs_end']: end}
      results_c.extend(s.run(graph.get_tensor_by_name('x_1/chars/MLP/Elu:0'),
                             feed_dict=feed))

    index_step = 0
    for batch_w in batches_w:
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
      feed = {ph['inputs_words']: batch_w, ph['loutw']: wout,
              ph['loutc']: cout}
      for a in s.run(graph.get_tensor_by_name('x_1/meta_char_word/ToInt32:0'),
                     feed_dict=feed):
        out_sentences.append([w for w in a])
    test_stringio_joint = tes.write_string(out_sentences)
    if not FLAGS.out:
      target_file = os.path.join(output_dir,
                                 os.path.basename(FLAGS.Test))
    else:
      target_file = FLAGS.out
    tes.write_stringio_to_file(target_file, test_stringio_joint)
    print('Wrote tagged file to %s' % target_file)
    #print('acc', tes.simple_eval(out_sentences))


def main(argv):
  run_testing()

if __name__ == '__main__':
  tf.app.run()
