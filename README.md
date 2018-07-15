# Software and hardware requirements

-   python 2.7
-   numpy
-   Tensorflow 1.5+
-   For fast training, a Nvidia graphic card or GPU

# Credits

This code is based on the paper: https://arxiv.org/abs/1805.08237 

Bernd Bohnet, Ryan McDonald, Gonçalo Simões, Daniel Andor, Emily Pitler, Joshua
Maynez. Morphosyntactic Tagging with a Meta-BiLSTM Model over Context Sensitive 
Token Encodings. ACL, 2018. 

Our tagger **ranked 1st** for morphological features in the CoNLL-2018 Shared Task and 
had strong results for many languages on upos tags. The tagger is especially strong
for cases where a wider context is required to determine the correct tag as for xpos 
and morphological features tagging.

Contributions:
Bernd Bohnet, Ryan McDonald, Gonçalo Simões, Daniel Andor, Emily Pitler, Joshua
Maynez, Terry Koo.

# Training a tagger

python train_cw.py --train='en-wsj-std-train-stanford-3.3.0.conll' \
--dev='en-wsj-std-dev-stanford-3.3.0.conll' \
--embeddings='glove.6B.100d.txt' \
--task='xtag' \
--config='config.json'

The paths need to be adapted. The 'config.json' file contains the settings 
for the hyperparamerters. The settings for the number of LSTM layers, 
cells, etc. are smaller than the sizes used in the paper.

The input and output files are in CoNLL-U format:
http://universaldependencies.org/format.html

The tagger supports three tasks: --task='upos' | 'xtag' | 'feats'


# Applying a tagger

python test_cw.py --test='en-wsj-std-test-stanford-3.3.0.conll' \
--task='xtag' \
--output_dir='model_save_dir' \
--out='output.conll'

