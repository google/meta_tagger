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

import io
import numpy as np
import tensorflow as tf



class Test(object):
  """Testing and simple evaluation of restuls."""

  def __init__(self, config, reader, filename, id_tag):
    self._config = config
    self._id_tag = id_tag
    test_data_org = reader.read_corpus(filename, [0, 1, 2, 3, 4, 5, 6, 7])
    self._test_sentences_org = reader.sentences(test_data_org)
    return

  def write_string(self, out_sentences):
    f = io.StringIO()
    for k in range(len(self._test_sentences_org)):
      for line_number, (e, p) in enumerate(
          zip(self._test_sentences_org[k], out_sentences[k]), 1):
        f.write(e[0])
        f.write(u'\t')
        f.write(e[1])
        f.write(u'\t')
        # lemma
        f.write(u'_')
        if self._config.tagging == 3:
          f.write(u'\t')
          f.write(self._id_tag[p])
          f.write(u'\t_\t_\t')
        elif self._config.tagging == 4:
          f.write(u'\t_\t')
          f.write(self._id_tag[p])
          f.write(u'\t_\t')
        elif self._config.tagging == 5:
          f.write(u'\t_\t_\t')
          f.write(self._id_tag[p])
          f.write(u'\t')
        f.write(str(1).decode('utf-8'))
        f.write(u'\t_\t_\n')
      f.write(u'\n')
    f.seek(0)
    return f

  def simple_eval(self, out_sentences):
    """Simple evaluation."""
    count = 0.0
    correct = 0.0
    for k in range(len(self._test_sentences_org)):
      for (e, p) in zip(self._test_sentences_org[k], out_sentences[k]):
        if self._id_tag[p] == str(e[self._config.tagging]):
          correct += 1
        count += 1
        #print(p, self._id_tag[p], e[2])
    return np.float32(correct / count)

  def write_string_aligned(self, out_sentences):
    """Aligns output with test input in case of removed multi word tokens."""

    f = io.StringIO()
    for sentence_index, snt_org in enumerate(self._test_sentences_org):
      token_index_sys = 0
      for cnt in range(len(snt_org)):
        p = out_sentences[sentence_index][token_index_sys]
        e = self._test_sentences_org[sentence_index][cnt]
        if u'-' in e[0]:
          f.write(e[0])
          f.write(u'\t')
          f.write(e[1])
          f.write(u'\t_\t_\t_\t_')
          f.write(u'\t_\t_\t_\t_')
          f.write(u'\n')
          continue

        token_index_sys += 1

        f.write(e[0])
        f.write(u'\t')
        f.write(e[1])
        f.write(u'\t')
        # lemma
        f.write(u'_')
        if self._config.tagging == 3:
          f.write(u'\t')
          f.write(self._id_tag[p])
          f.write(u'\t_\t_\t')
        elif self._config.tagging == 4:
          f.write(u'\t_\t')
          f.write(self._id_tag[p])
          f.write(u'\t_\t')
        elif self._config.tagging == 5:
          f.write(u'\t_\t_\t')
          f.write(self._id_tag[p])
          f.write(u'\t')
        f.write(str(1).decode('utf-8'))
        f.write(u'\t_\t_\t_')
        f.write(u'\n')
      f.write(u'\n')
    f.seek(0)
    return f

  def write_stringio_to_file(self, filename, stringio):
    """Writes stringio object to a file.

    Args:
      filename: path and file name.
      stringio: stringio output file.
    """
    stringio.seek(0)
    with tf.gfile.GFile(filename, 'w') as f:
      for line in stringio:
        f.write(line)
        #f.write(u'\n')
