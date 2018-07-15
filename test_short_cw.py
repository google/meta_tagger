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
"""Applies the tagger and supposed to standalone."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import os
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', '~/tensorboard/wsj_xtag_v2/', '')
flags.DEFINE_string('task_name', 'meta_lstm_v6_e4', '')
flags.DEFINE_string('testing_data', 'en-wsj-std-test-stanford-3.3.0.conll', '')


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
    print(sstart)
    print(send)
    print(sntc)
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


def main(_):
  hparams = tf.contrib.training.HParams() # task_name='meta_word_char_v2'
  output_dir = os.path.expanduser(FLAGS.output_dir)
  voc = Vocab(output_dir)

  (batches_w, batches_c, batches_start,
   batches_end) = read_corpus(FLAGS.testing_data, voc.char_id, voc.word_id,
                              voc.pred_id)
  print('Warning: batching is not used and cpu, hence tagging is slow.')
  with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as s:

    filename_model = os.path.join(output_dir, FLAGS.task_name)
    print('filename_model', filename_model)

    saver = tf.train.import_meta_graph(filename_model + '.meta')
    s.run(tf.global_variables_initializer())
    saver.restore(s, filename_model)
    ph_names = ['inputs_words', 'inputs_chars', 'indexs_start', 'indexs_end',
                'loutw', 'loutc']
    graph = tf.get_default_graph()
    ph = {name: graph.get_tensor_by_name(name + ':0') for name in ph_names}

    out_sentences = []
    for (cnt, (batch_w, batch_c, start, end)) in enumerate(
        zip(batches_w, batches_c, batches_start, batches_end), 1):
      feed = {ph['inputs_words']: batch_w, ph['inputs_chars']: batch_c,
              ph['indexs_start']: start, ph['indexs_end']: end}
      wout, cout = s.run([graph.get_tensor_by_name('x_1/words/MLP/Elu:0'),
                          graph.get_tensor_by_name('x_1/chars/MLP/Elu:0')],
                         feed_dict=feed)
      if not cnt % 100: print('sentence', cnt)
      feed = {ph['inputs_words']: batch_w, ph['loutw']: wout, ph['loutc']: cout}
      a = s.run(graph.get_tensor_by_name('x_1/meta_char_word/ToInt32:0'),
                feed_dict=feed)
      out_sentences.append([voc.id_tag[w] for w in a[0]])
    for sentence in out_sentences:
      for tag in sentence:
        print (tag)

if __name__ == '__main__':
  tf.app.run()
