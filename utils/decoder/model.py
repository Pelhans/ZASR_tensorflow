"""Contains DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import logging
import gzip
import copy
import numpy as np
import inspect
from decoders.swig_wrapper import Scorer
from decoders.swig_wrapper import ctc_greedy_decoder
from decoders.swig_wrapper import ctc_beam_search_decoder_batch


class lm_decoder(object):

    def __init__(self, beam_alpha, beam_beta, language_model_path,
                        vocab_list):
        """Initialize the external scorer.

        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param language_model_path: Filepath for language model. If it is
                                    empty, the external scorer will be set to
                                    None, and the decoding method will be pure
                                    beam search without scorer.
        :type language_model_path: basestring|None
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        """
        if language_model_path != '':
            print("begin to initialize the external scorer "
                             "for decoding")
            self._ext_scorer = Scorer(beam_alpha, beam_beta,
                                      language_model_path, vocab_list)
            lm_char_based = self._ext_scorer.is_character_based()
            lm_max_order = self._ext_scorer.get_max_order()
            lm_dict_size = self._ext_scorer.get_dict_size()
            print("language model: "
                             "is_character_based = %d," % lm_char_based +
                             " max_order = %d," % lm_max_order +
                             " dict_size = %d" % lm_dict_size)
            print("end initializing scorer")
        else:
            self._ext_scorer = None
            print("no language model provided, "
                             "decoding by pure beam search without scorer.")

    def decode_batch_beam_search(self, probs_split, beam_alpha, beam_beta,
                                 beam_size, cutoff_prob, cutoff_top_n,
                                 vocab_list, num_processes):
        """Decode by beam search for a batch of probs matrix input.

        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                        characters with highest probs in vocabulary will be
                        used in beam search, default 40.
        :type cutoff_top_n: int
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of basestring
        """
        if self._ext_scorer != None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)
        # beam search decode
        print ("probs_split: ", np.shape(probs_split), probs_split)
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self._ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n)

        results = [result[0][1] for result in beam_search_results]
        return results

    def _adapt_feeding_dict(self, feeding_dict):
        """Adapt feeding dict according to network struct.

        To remove impacts from padding part, we add scale_sub_region layer and
        sub_seq layer. For sub_seq layer, 'sequence_offset' and
        'sequence_length' fields are appended. For each scale_sub_region layer
        'convN_index_range' field is appended.

        :param feeding_dict: Feeding is a map of field name and tuple index
                             of the data that reader returns.
        :type feeding_dict: dict|list
        :return: Adapted feeding dict.
        :rtype: dict|list
        """
        adapted_feeding_dict = copy.deepcopy(feeding_dict)
        if isinstance(feeding_dict, dict):
            adapted_feeding_dict["sequence_offset"] = len(adapted_feeding_dict)
            adapted_feeding_dict["sequence_length"] = len(adapted_feeding_dict)
            for i in xrange(self._num_conv_layers):
                adapted_feeding_dict["conv%d_index_range" %i] = \
                        len(adapted_feeding_dict)
        elif isinstance(feeding_dict, list):
            adapted_feeding_dict.append("sequence_offset")
            adapted_feeding_dict.append("sequence_length")
            for i in xrange(self._num_conv_layers):
                adapted_feeding_dict.append("conv%d_index_range" % i)
        else:
            raise ValueError("Type of feeding_dict is %s, not supported." %
                             type(feeding_dict))

        return adapted_feeding_dict

    def _adapt_data(self, data):
        """Adapt data according to network struct.

        For each convolution layer in the conv_group, to remove impacts from
        padding data, we can multiply zero to the padding part of the outputs
        of each batch normalization layer. We add a scale_sub_region layer after
        each batch normalization layer to reset the padding data.
        For rnn layers, to remove impacts from padding data, we can truncate the
        padding part before output data feeded into the first rnn layer. We use
        sub_seq layer to achieve this.

        :param data: Data from data_provider.
        :type data: list|function
        :return: Adapted data.
        :rtype: list|function
        """

        def adapt_instance(instance):
            if len(instance) < 2 or len(instance) > 3:
                raise ValueError("Size of instance should be 2 or 3.")
            padded_audio = instance[0]
            text = instance[1]
            # no padding part
            if len(instance) == 2:
                audio_len = padded_audio.shape[1]
            else:
                audio_len = instance[2]
            adapted_instance = [padded_audio, text]
            # Stride size for conv0 is (3, 2)
            # Stride size for conv1 to convN is (1, 2)
            # Same as the network, hard-coded here
            padded_conv0_h = (padded_audio.shape[0] - 1) // 2 + 1
            padded_conv0_w = (padded_audio.shape[1] - 1) // 3 + 1
            valid_w = (audio_len - 1) // 3 + 1
            adapted_instance += [
                [0],  # sequence offset, always 0
                [valid_w],  # valid sequence length
                # Index ranges for channel, height and width
                # Please refer scale_sub_region layer to see details
                [1, 32, 1, padded_conv0_h, valid_w + 1, padded_conv0_w]
            ]
            pre_padded_h = padded_conv0_h
            for i in xrange(self._num_conv_layers - 1):
                padded_h = (pre_padded_h - 1) // 2 + 1
                pre_padded_h = padded_h
                adapted_instance += [
                    [1, 32, 1, padded_h, valid_w + 1, padded_conv0_w]
                ]
            return adapted_instance

        if isinstance(data, list):
            return map(adapt_instance, data)
        elif inspect.isgeneratorfunction(data):

            def adapted_reader():
                for instance in data():
                    yield map(adapt_instance, instance)

            return adapted_reader
        else:
            raise ValueError("Type of data is %s, not supported." % type(data))

    def _create_parameters(self, model_path=None):
        """Load or create model parameters."""
        if model_path is None:
            self._parameters = paddle.parameters.create(self._loss)
        else:
            self._parameters = paddle.parameters.Parameters.from_tar(
                gzip.open(model_path))

    def _create_network(self, vocab_size, num_conv_layers, num_rnn_layers,
                        rnn_layer_size, use_gru, share_rnn_weights):
        """Create data layers and model network."""
        # paddle.data_type.dense_array is used for variable batch input.
        # The size 161 * 161 is only an placeholder value and the real shape
        # of input batch data will be induced during training.
        audio_data = paddle.layer.data(
            name="audio_spectrogram",
            type=paddle.data_type.dense_array(161 * 161))
        text_data = paddle.layer.data(
            name="transcript_text",
            type=paddle.data_type.integer_value_sequence(vocab_size))
        seq_offset_data = paddle.layer.data(
            name='sequence_offset',
            type=paddle.data_type.integer_value_sequence(1))
        seq_len_data = paddle.layer.data(
            name='sequence_length',
            type=paddle.data_type.integer_value_sequence(1))
        index_range_datas = []
        for i in xrange(num_rnn_layers):
            index_range_datas.append(
                paddle.layer.data(
                    name='conv%d_index_range' % i,
                    type=paddle.data_type.dense_vector(6)))

        self._log_probs, self._loss = deep_speech_v2_network(
            audio_data=audio_data,
            text_data=text_data,
            seq_offset_data=seq_offset_data,
            seq_len_data=seq_len_data,
            index_range_datas=index_range_datas,
            dict_size=vocab_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_layer_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)
