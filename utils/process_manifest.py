#!/usr/bin/env python
# coding=utf-8

import json
from collections import Counter

def get_path_trans(manifest_path="data/aishell/manifest.train"  ):
    '''
    Get path_to_wav and transcript list 
    from data/manifest.{train,dev,test}
    '''

    path_to_wav = []
    transcript = []
    duration = []
    lines = open(manifest_path, "r").readlines()
    for line in lines:
        man_dict = json.loads(line)
        path_to_wav.append(man_dict["audio_filepath"])
        transcript.append(man_dict["text"])
        duration.append(man_dict["duration"])
    return path_to_wav, transcript, duration

def create_dict(vocab):
    '''
    Creat word dict and map from word to num
    '''

    total_words = open(vocab, 'r').readlines()
    total_words = [word.strip() for word in total_words]
    counter = Counter(total_words)
    words = sorted(counter)
    word_size = len(words)
    word_num_map = dict(zip(words, range(word_size)))
    
    print "word_size: ", word_size
    return word_size, words, word_num_map

if __name__ == "__main__":
    get_path_trans("../data/aishell/manifest.test")
    creat_dict("../data/aishell/vocab.txt")
