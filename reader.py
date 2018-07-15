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
import collections
import io
import os
import numpy as np
import tensorflow as tf

logging = tf.logging

PAD = 0
UNK = 1


def read_file_to_stringio(filename):
  """Reads a file into a unicode io.StringIO for faster IO."""
  string_io = io.StringIO()
  with tf.gfile.GFile(filename, 'r') as f:
    for line in codecs.getreader('utf-8')(f, errors='ignore'):
      string_io.write(line)

  string_io.seek(0)
  return string_io


class Reader(object):
  """Reading and batching of data."""

  def __init__(self, config, delimiter='\t'):

    # PAD and UNK
    self.special_tokens = [u'<P>', u'<U>']
    self.config = config
    self.delimiter = delimiter

    # Maps words to indexes for the embeddings.
    self.embedding_dict = {}
    self.embeddings = {}

  def read_corpus(self, filename, indices):
    """Reads a corpus in CoNLL format.

    Args:
      filename: Path and file name for the corpus.
      indices: Conll file column indices.

    Returns:
      A list of the lines of the corpus. The input is split again
      into lists that contain the content of the lines. E.g.
      [['The' 'DT']
       ['piano' 'NN' '0']
       []
      ]
    """
    filename = os.path.expanduser(filename)
    data = []
    with tf.gfile.GFile(filename, 'r') as f:
      for line in codecs.getreader('utf-8')(f, errors='ignore'):
        if line.strip().startswith(u'#'):
          continue
        split = np.array(line.split(u'\t'))
        if len(split) > 4:
          data.append(split[indices])
        else:
          # Avoid splits with one element such as line breaks '\n'.
          data.append(np.array([]))
    return data

  def to_char_corpus(self, corpus, tag_position=1):
    """Converting a read corpus to a char corpus.

    Args:
      filename: Path and file name for the corpus.
      tag_position: Position of the tag.

    Args:
     inputs ...
      [['The' 'DT']
       ['piano' 'NN']
       []
      ]
    Returns:
     [[[T DT] [h DT] [e DT] [' ' S] [p NN] ... ]
     ]
    """

    char_corpus = []
    sentence = []
    total = 0
    for entry in corpus:
      if len(entry) > 0:
        word = entry[0]
        tag = entry[tag_position]
        chars = list(word)
        for c in chars:
          total += 1
          sentence.append([c, tag])
        sentence.append([u'\t', u'SP'])
        total += 1
      else:
        char_corpus.append(sentence)
        sentence = []
    return char_corpus

  def elements_to_dict(self, corpus, element_index, lowercase=False):
    """Builds a dictionary with ids."""

    if lowercase and self.config.lowercase:
      counter = collections.Counter(
          e[element_index].lower() for s in corpus for e in s)
    else:
      counter = collections.Counter(e[element_index] for s in corpus for e in s)

    # Remove words that that are less frequent as in min_occurrence.
    for w in list(counter.keys()):
      if counter[w] < self.config.min_occurrence:
        del counter[w]
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    elements, _ = list(zip(*count_pairs))
    element_id = dict(
        zip(elements,
            range(
                len(self.special_tokens),
                len(elements) + len(self.special_tokens))))
    for index, symbol in enumerate(self.special_tokens, 0):
      element_id[symbol] = index
    return element_id

  def word_list_counts(self, corpora, element_index):
    """Builds a list with word counts from a corporas."""
    counter = collections.Counter()
    for corpus in corpora:
      for s in corpus:
        for e in s:
          word = e[element_index]
          if word in counter:
            counter[word] += 1
          else:
            counter[word] = 1

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return count_pairs

  def build_word_vocab(self, corpus, lowercase):
    """Builds char and tag vocab dictionaries."""

    words_id = self.elements_to_dict(corpus, 0, lowercase=lowercase)
    return words_id

  def build_char_vocab(self, char_corpus, add_special_tokens=False):
    """Builds char and tag vocab dictionaries."""

    chars_id = self.elements_to_dict(char_corpus, 0)
    if add_special_tokens:
      keys = chars_id.keys()
      values = chars_id.values()
      next_index = max(values) + 1
      for tok in self.special_tokens:
        for c in tok:
          if c not in keys:
            chars_id[c] = next_index
            keys.append(c)
            next_index += 1

    tags_id = self.elements_to_dict(char_corpus, 1)
    return chars_id, tags_id

  def _convert_ids(self, sentence, word_to_id, tag_to_id, rel_to_id):
    """Converts words and tags of a sentence into ids."""

    sentence_id = []
    for (w, t, h, l) in sentence:
      word_id = UNK
      if self.config.lowercase:
        w = w.lower()
      if w in word_to_id:
        word_id = word_to_id[w]

      pre_word_id = UNK
      if w in self.embedding_dict:
        pre_word_id = self.embedding_dict[w]
      tag_id = UNK
      if t in tag_to_id:
        tag_id = tag_to_id[t]
      if h == '_':
        head = 0
      else:
        head = int(h)
      sentence_id.append([word_id, pre_word_id, tag_id, head, rel_to_id[l]])
    return sentence_id

  def _sentences(self, data, word_to_id, tag_to_id, rel_to_id):
    """Converts input data to list of sentences."""

    sentence = []
    sentences = []
    for l in data:
      if len(l) != 0:
        sentence.append(l)
      else:
        snt = self._convert_ids(sentence, word_to_id, tag_to_id, rel_to_id)
        sentences.append(snt)
        sentence = []
    if len(l) != 0:
      sentences.append(sentence)
    return sentences

  def sentence_ids(self, sentence, word_id, tag_id, pred_id=None):
    sentence_id = []
    for token in sentence:
      word, tag = token
      if self.config.lowercase:
        word = word.lower()
      w_id = word_id.get(word, UNK)
      if pred_id:
        p_id = pred_id.get(word, UNK)
      else:
        p_id = self.embedding_dict.get(word, UNK)
      t_id = tag_id.get(tag, UNK)
      sentence_id.append([w_id, p_id, t_id])
    return sentence_id

  def sentences_ids(self, data, word_id, tag_id, pred_id=None):
    """Converts input data to list of sentences."""

    sentence = []
    sentences = []
    for l in data:
      if len(l) != 0:
        sentence.append(l)
      else:
        snt = self.sentence_ids(sentence, word_id, tag_id, pred_id)
        sentences.append(snt)
        sentence = []
    if len(l) != 0:
      sentences.append(sentence)
    return sentences

  def check_for_unknown_pretrained_embeddings(self, sentences, word_id_map):
    total = 0
    unk_word = 0
    unk_pre = 0
    pre = 0
    unk_pre_word_kn = 0
    for snt in sentences:
      for entry in snt:
        total += 1
        word = entry[0]
        pretrained_word_id = entry[1]
        if word == UNK:
          unk_word += 1
        if pretrained_word_id == UNK:
          unk_pre += 1
        if pretrained_word_id == UNK and word != UNK:
          unk_pre_word_kn += 1

    words = word_id_map.keys()
    lowercase_words = 0
    contains_upper_words = 0
    numbers = 0
    for word in words:
      if word.islower():
        lowercase_words += 1
      else:
        contains_upper_words += 1
        if word.isdigit():
          numbers += 1

    logging.info('total words in corpus %d', total)
    logging.info('words in corpus marked unknown (word == UNK) %d', unk_word)
    logging.info('pretrained words unknown (pretrain_word_id == UNK) %d',
                 unk_pre)
    logging.info(
        'words not found in pretrained but known words,'
        'unk_pre_word_kn (pretrained_word_id == UNK and '
        'word != UNK) %d', unk_pre_word_kn)
    logging.info('word-index mapping, total keys %d cased %d lowercased %d',
                 len(words), contains_upper_words, lowercase_words)
    logging.info('The cased words might be or contain numbers (isdigit()) %d',
                 numbers)

  def sentences(self, data):
    """Converts input data to list of sentences."""

    sentence = []
    sentences = []
    for l in data:
      if len(l) != 0:
        sentence.append(l)
      else:
        sentences.append(sentence)
        sentence = []
    if len(l) and len(sentence) != 0:
      sentences.append(sentence)
    return sentences

  def char_sentences(self,
                     corpus,
                     char_id,
                     tag_id,
                     remove_annotation=False,
                     separator=-1):
    """Converts char sentences to list of sentences.

       [[[T DT The] [h DT] [e DT] [' ' SP] .. ]]
       to
       [[[29 7] [11 7] .. ]]]
    """
    if remove_annotation:
      print('remove_annotation!!!')
      exit()
    sentences = []
    for sentence in corpus:
      sentence_id = []
      for c, t in sentence:
        c_id = char_id.get(c, UNK)
        t_id = tag_id.get(t, UNK)

        # Remove annotation (e.g. for testing).
        if remove_annotation and t_id != separator:
          t_id = UNK
        sentence_id.append([c_id, t_id])

      sentences.append(sentence_id)
    return sentences

  def sentences_to_buckets(self, hparams, sentences):
    """Converts list of sentences in to a list of batches.

    Args:
      hparams: Parameters such as the batch size.
      sentences: Sentences to batch.

    Returns:
      List of batches
    """
    # Extract input and target values
    inputs = [[entry[0:3] for entry in snt] for snt in sentences]

    # Build batches with buckets.
    k = 0
    used_batch_size = 0
    start_sentence = 0
    buckets = 1
    batches = 0
    input_batch = []
    input_batches = []
    largest = 0
    while k < len(inputs):
      expand_ok = True

      if len(inputs[k]) > largest:
        if (buckets * len(inputs[k])) <= hparams.batch_word_size:
          largest = len(inputs[k])
        else:
          expand_ok = False

      used = buckets * largest

      # Extend bucket if it does not exceed batch size.
      if used <= hparams.batch_word_size and expand_ok:
        input_batch.append(inputs[k])
      else:
        # Batch got too large, then remove it and start a new batch.
        # Used batch size: usage = (buckets - 1) * len(inputs[k - 1])

        bucket_size = largest
        for snt in input_batch:
          length_diff = bucket_size - len(snt)
          if length_diff > 0:
            for _ in range(0, length_diff):

              # Add a entry with 'PAD' symbols.
              snt.append([PAD for _ in snt[0]])

        # Add the batch to the batches.
        input_batches.append(input_batch)

        # Reset and prepare to collect next batch.
        largest = 1
        start_sentence = k
        used_batch_size = 0
        buckets = 1
        batches += batches
        k -= 1
        input_batch = []

      used_batch_size += len(inputs[k])
      buckets += 1
      k += 1

    if k > start_sentence:
      bucket_size = largest  # len(input_batch[-1])
      for snt in input_batch:
        length_diff = bucket_size - len(snt)
        if length_diff > 0:
          for i in range(0, length_diff):

            # Add a entry with 'PAD' symbols.
            snt.append([PAD for i in snt[0]])
      input_batches.append(input_batch)

    logging.info('Created %d word batches.' % len(input_batches))
    np_in_batches = np.asarray(input_batches)
    return np_in_batches

  def char_sentences_to_buckets_index_sc(self, config, sentences, sp_char):
    """Converts list of sentences into list of batches."""
    # Extract input and target values
    inputs = [[entry[0:2] for entry in snt] for snt in sentences]

    # Build batches with buckets.
    k = 0
    used_batch_size = 0
    start_sentence = 0
    buckets = 1
    batches = 0

    # index into
    input_index = []
    input_index_start = []

    input_batch = []
    input_index_batch = []
    input_index_batch_start = []
    input_batches = []
    largest = 0
    largest_word_index = 0

    tagets_batches = []
    target_batch = []

    # provides the index of the sentence in a bucket
    snt_count = 0
    while k < len(inputs):
      expand_ok = True

      # input sentences larger then pat largest sentence plus one pad
      if (len(inputs[k]) + 1) > largest:
        if (buckets * len(inputs[k])) <= config.batch_char_size:
          largest = len(inputs[k]) + 1
        else:
          expand_ok = False
      used = buckets * largest

      # Extend bucket if it does not exceed batch size.
      if used <= config.batch_char_size and expand_ok:
        sntx = []
        for tok in inputs[k]:
          sntx.append(tok[0])
        input_batch.append(sntx)

        # Gather the start and end index of the sentences
        snt_word_indices_start = []
        snt_word_indices = []
        snt_word_indices_start.append([snt_count, 0])
        snt_target = []
        snt_target.append(inputs[k][0][1])

        last_was_sp_char = False
        for char_index, tt in enumerate(inputs[k]):
          if last_was_sp_char:
            snt_word_indices_start.append([snt_count, char_index])
            snt_target.append(tt[1])
          if tt[0] == sp_char:
            snt_word_indices.append([snt_count, char_index - 1])  # -1
            last_was_sp_char = True
          else:
            last_was_sp_char = False

        input_index.append(snt_word_indices)
        input_index_start.append(snt_word_indices_start)
        target_batch.append(snt_target)
        snt_count += 1

        # Find the last word + 1 for the end.
        if (len(snt_word_indices) + 1) > largest_word_index:
          largest_word_index = len(snt_word_indices) + 1
      else:
        k -= 1

      if (used > config.batch_char_size) or not expand_ok:
        # Start a new batch.

        # Used batch size: usage = (buckets - 1) * largest  #len(inputs[k - 1])

        # Pad the sentences.
        bucket_size = largest
        for snt in input_batch:
          length_diff = bucket_size - len(snt)
          if length_diff > 0:
            for _ in range(0, length_diff):

              # Add a entry with 'PAD' symbols.
              snt.append(PAD)

        bucket_size = largest_word_index
        snt_count = 0
        # Pad the end index.
        last_index = 0
        #for snt in input_index:
        for end, snt, trg in zip(input_index, input_index_start, target_batch):
          length_diff = bucket_size - len(snt)
          last_index = end[-1][1] + 2
          if length_diff > 0:
            for _ in range(0, length_diff):

              # Add a entry
              #snt.append([snt_count, last_index])
              end.append([snt_count, last_index])
              snt.append([snt_count, last_index])
              trg.append(PAD)

          snt_count += 1

        bucket_size = largest_word_index

        # Add the batch to the batches.
        input_batches.append(input_batch)
        input_batch = []

        input_index_batch.append(input_index)
        input_index_batch_start.append(input_index_start)
        input_index = []
        input_index_start = []

        tagets_batches.append(target_batch)
        target_batch = []

        # Reset and prepare to collect next batch.
        start_sentence = k + 1
        used_batch_size = 0
        buckets = 1
        batches += 1
        largest = 0
        largest_word_index = 0
        snt_count = 0

      used_batch_size += len(inputs[k])
      buckets += 1
      k += 1

    if k > start_sentence:
      bucket_size = largest  # len(input_batch[-1])
      for snt in input_batch:
        length_diff = bucket_size - len(snt)
        if length_diff > 0:
          for unused_i in range(0, length_diff):

            # Add a entry with 'PAD' symbols.
            snt.append(PAD)
      input_batches.append(input_batch)

    if k > start_sentence:
      bucket_size = largest_word_index
      snt_count = 0
      last_index = 0
      for end, snt, trg in zip(input_index, input_index_start, target_batch):
        length_diff = bucket_size - len(snt)
        last_index = end[-1][1] + 2
        if length_diff > 0:
          for unused_i in range(0, length_diff):

            # Add a entry with 'PAD' symbols.
            end.append([snt_count, last_index])
            snt.append([snt_count, last_index])
            trg.append(PAD)
        snt_count += 1
      input_index_batch.append(input_index)


      input_index_batch_start.append(input_index_start)
      tagets_batches.append(target_batch)

    np_in_batches = np.asarray(input_batches)
    logging.info('Created %d char batches.' % len(np_in_batches))

    return (np_in_batches, input_index_batch, input_index_batch_start,
            tagets_batches)

  def load(self, filename):
    """Load Embeddings."""
    filename = os.path.expanduser(filename)
    embeddings = []
    self.embedding_dict = {}
    cur_idx = len(self.special_tokens)
    open_func = tf.gfile.GFile
    with open_func(filename, 'rb') as f:
      reader = codecs.getreader('utf-8')(f, errors='ignore')
      for _, line in enumerate(reader):
        line = line.rstrip().split(' ')
        if len(line) > 2:
          embeddings.append(np.array(line[1:], dtype=np.float32))
          self.embedding_dict[line[0]] = cur_idx
          cur_idx += 1
    embeddings = np.stack(embeddings)
    embeddings = np.pad(embeddings, ((len(self.special_tokens), 0), (0, 0)),
                        'constant')
    self.embeddings = np.stack(embeddings)
    self.embed_size = embeddings.shape[1]
    logging.info('embeddings size %d' % self.embed_size)
    return
