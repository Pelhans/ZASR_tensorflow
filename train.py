#!/usr/bin/env python
# coding=utf-8

import numpy as np                   
import tensorflow as tf              
from tensorflow.python.ops import ctc_ops

from utils import process_manifest
from model_utils import init_model
from conf import hyparam, config

hyparam  = hyparam.Config()
conf = config.Config()

wav_files, text_labels, _ = process_manifest.get_path_trans()

words_size, words, word_num_map = process_manifest.create_dict("data/aishell/vocab.txt")


bi_rnn = init_model.BiRNN(wav_files, text_labels, words_size, words, word_num_map)

bi_rnn.build_train()
