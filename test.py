#!/usr/bin/env python
# coding=utf-8

import os

from data_utils import process_manifest
from model_utils import init_model
from conf.hyparam import Config

conf = Config()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

wav_files, text_labels, _ = process_manifest.get_path_trans()

words_size, words, word_num_map = process_manifest.create_dict(conf.vocab_path)

deepspeech2 = init_model.DeepSpeech2(wav_files, text_labels, words_size, words, word_num_map)

deepspeech2.build_test()
