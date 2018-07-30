#!/usr/bin/env python
# coding=utf-8

import os

from utils import process_manifest
from model_utils import init_model
from conf import hyparam, config

hyparam  = hyparam.Config()
conf = config.Config()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

wav_files, text_labels, _ = process_manifest.get_path_trans()

words_size, words, word_num_map = process_manifest.create_dict("data/aishell/vocab.txt")

deepspeech2 = init_model.DeepSpeech2(wav_files, text_labels, words_size, words, word_num_map)
deepspeech2.build_test()
