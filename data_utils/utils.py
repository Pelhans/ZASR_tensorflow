# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
from collections import Counter

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from data_utils.audio_featurizer import AudioFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer
from conf.hyparam import Config

'''
To help creat train set and get batch from data
'''

def next_batch(start_idx=0,
               batch_size=1,
               n_input=None,
               n_context=None,
               labels=None,
               wav_files=None,
               word_num_map=None,
               specgram_type='mfcc'):
    """ Get data batch for training
    
    :param start_idx:
    :param batch_size:
    :param n_input:
    :param n_context:
    :param labels:
    :param wav_files:
    :param word_num_map:
    :param specgram_type
    :return:
    """
    filesize = len(labels)
    end_idx = min(filesize, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)
    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    audio_features, audio_features_len, text_vector, text_vector_len = get_audio_mfcc_features(None,
                                                                                               wav_files,
                                                                                               n_input,
                                                                                               n_context,
                                                                                               word_num_map,
                                                                                               txt_labels,
                                                                                               specgram_type)

    start_idx += batch_size
    # confirm start_idx
    if start_idx >= filesize:
        start_idx = -1

    # use 0 padding when serveral inputs
    audio_features, audio_features_len = pad_sequences(audio_features)
    sparse_labels = sparse_tuple_from(text_vector)

    return start_idx, audio_features, audio_features_len, sparse_labels, wav_files


def get_audio_mfcc_features(txt_files, wav_files, n_input,
                            n_context, word_num_map, txt_labels=None, 
                            specgram_type='mfcc', mean_std_filepath='data/aishell/mean_std.npz'):
    """ Get MFCC/linear specgram  features. The dim of MFCC is 39, contains 13 mfcc + 13 delta1 + 13 delta2.
        Linear specgram contains 161 features in different frequency section.
    
    :param txt_files:
    :param wav_files:
    :param n_input:
    :param n_context:
    :param word_num_map:
    :param txt_labels:
    :return:
    """
    audio_features = []
    audio_features_len = []
    text_vector = []
    text_vector_len = []
    if txt_files != None:
        txt_labels = txt_files
    get_feature = AudioFeaturizer(specgram_type)
    normalizer = FeatureNormalizer(mean_std_filepath)
    for txt_obj, wav_file in zip(txt_labels, wav_files):
        # Turn inputs into features
        if specgram_type == 'mfcc':
            audio_data = audiofile_to_input_vector(wav_file, n_input, n_context) # get mfcc feature ( ???, 741 )
        elif specgram_type == 'linear':
            speech_segment = SpeechSegment.from_file(wav_file, "")
            specgram = get_feature.featurize(speech_segment)
            audio_data = normalizer.apply(specgram)
            audio_data = np.transpose(audio_data) # get linear specgram feature, (?, 161)
        audio_data = audio_data.astype('float32')

        audio_features.append(audio_data)
        audio_features_len.append(np.int32(len(audio_data)))

        target = []
        if txt_files != None:  # txt_obj是文件
            target = trans_text_ch_to_vector(txt_obj, word_num_map)
        else:
            target = trans_text_ch_to_vector(None, word_num_map, txt_obj)  # txt_obj是labels
        text_vector.append(target)
        text_vector_len.append(len(target))

    audio_features = np.asarray(audio_features)
    audio_features_len = np.asarray(audio_features_len)
    text_vector = np.asarray(text_vector)
    text_vector_len = np.asarray(text_vector_len)
    return audio_features, audio_features_len, text_vector, text_vector_len


def sparse_tuple_from(sequences, dtype=np.int32):
    """ Turn dense matrix to sparse matrix

    :param sequences:
    :param dtype:
    :return:
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def trans_text_ch_to_vector(txt_file, word_num_map, txt_label=None):
    """ Trans chinese chars to vector
    
    :param txt_file:
    :param word_num_map:
    :param txt_label:
    :return:
    """
    words_size = len(word_num_map)

    to_num = lambda word: word_num_map.get(word.encode('utf-8'), words_size)

    if txt_file != None:
        txt_label = get_ch_lable(txt_file)

    labels_vector = list(map(to_num, txt_label))
    return labels_vector


def get_ch_lable(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('gb2312')
    return labels


def trans_tuple_to_texts_ch(tuple, words):
    """ Trans vector to chars
    
    :param tuple:
    :param words:
    :return:
    """
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == 0 else words[c]  # chr(c + FIRST_INDEX)
        results[index] = results[index] + c

    return results


def trans_array_to_text_ch(value, words):
    results = ''
    for i in range(len(value)):
        results += words[value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')


def audiofile_to_input_vector(audio_filename, n_input, n_context):
    """ Compute MFCC features with n_context

    :param audio_filename:
    :param n_input:
    :param n_context:
    :return:
    """
    fs, audio = wav.read(audio_filename)

    # get mfcc features with dim 39
    get_feature = AudioFeaturizer("mfcc")
    speech_segment = SpeechSegment.from_file(audio_filename, "")
    orig_inputs = get_feature.featurize(speech_segment) # (39, ?)
    orig_inputs = np.transpose(orig_inputs) # trans to time major (?, 39)


    train_inputs = np.zeros((orig_inputs.shape[0], n_input + 2 * n_input * n_context)) #(***/2, 195)
    empty_mfcc = np.zeros((n_input))

    # Prepare input data, consist of three parts, 
    # output is (past hyparam.n_context * 39 + current + future hyparam.n_context * 39)
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + n_context
    context_future_max = time_slices[-1] - n_context
    for time_slice in time_slices:
        # padding with 0 for the first of 9，mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[ max(0, time_slice - n_context):time_slice]

        # padding with 0 for the last of 9，mfcc features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + n_context + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, n_context * 39)
        now = orig_inputs[time_slice]
        future = np.reshape(future, n_context * n_input)
        train_inputs[time_slice] = np.concatenate((past, now, future))

    # Tran data to Norm distribution, minus mean value then over the varr
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    # shape of train_inputs: (shape(orig_inputs)/2, n_context * 2 * 39 + 39)
    return train_inputs 


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    """ Padding data with 0
    
    :param sequences:
    :param maxlen:
    :param dtype:
    :param padding:
    :param truncating:
    :param value:
    :return:
    """
    sequences_each_len = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(sequences_each_len)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, sequences_each_len


if __name__ == "__main__":

    print("")
