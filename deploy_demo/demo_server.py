"""Server-end for the ASR demo."""
import os
import time
import random
import re
import argparse
import functools
from time import gmtime, strftime
import SocketServer
import struct
import wave

import _init_paths
import assert_zero
from data_utils import process_manifest
from model_utils import init_model
from conf.hyparam import Config
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
conf = Config()
# yapf: disable
add_arg('host_port',        int,    8086,    "Server's IP port.")
add_arg('beam_size',        int,    conf.beam_size,    "Beam search width.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.5,   "Coef of LM for beam search.")
add_arg('beta',             float,  0.3,   "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  0.99,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   True,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   False,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('host_ip',          str,
        'localhost',
        "Server's IP address.")
add_arg('speech_save_dir',  str,
        'demo_cache',
        "Directory to save demo audios.")
add_arg('warmup_manifest',  str,
        'data/aishell/manifest.dev',
        "Filepath of manifest to warm up.")
add_arg('mean_std_path',    str,
        'data/potevio/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/potevio/vocab.txt',
        "Filepath of vocabulary.")
add_arg('model_path',       str,
        './checkpoints/potevio/params.tar.gz',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('lang_model_path',  str,
        'models/lm/zh_giga.no_cna_cmn.prune01244.klm',
        "Filepath for language model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()


class AsrTCPServer(SocketServer.TCPServer):
    """The ASR TCP Server."""

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 speech_save_dir,
                 audio_process_handler,
                 bind_and_activate=True):
        self.speech_save_dir = speech_save_dir
        self.audio_process_handler = audio_process_handler
        SocketServer.TCPServer.__init__(
            self, server_address, RequestHandlerClass, bind_and_activate=True)


class AsrRequestHandler(SocketServer.BaseRequestHandler):
    """The ASR request handler."""

    def handle(self):
        # receive data through TCP socket
        chunk = self.request.recv(1024)
        target_len = struct.unpack('>i', chunk[:4])[0]
        data = chunk[4:]
        while len(data) < target_len:
            chunk = self.request.recv(1024)
            data += chunk
        # write to file
        filename = self._write_to_file(data)

        print("Received utterance[length=%d] from %s, saved to %s." %
              (len(data), self.client_address[0], filename))
        start_time = time.time()
        transcript = self.server.audio_process_handler(filename)
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))
        if transcript == None:
            transcript = ""
        self.request.sendall(transcript)

    def _write_to_file(self, data):
        # prepare save dir and filename
        if not os.path.exists(self.server.speech_save_dir):
            os.mkdir(self.server.speech_save_dir)
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        out_filename = os.path.join(
            self.server.speech_save_dir,
            timestamp + "_" + self.client_address[0] + ".wav")
        # write to wav file
        file = wave.open(out_filename, 'wb')
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(data)
        file.close()
        return out_filename


def warm_up_test(audio_process_handler,
                 manifest_path,
                 num_test_cases,
                 random_seed=0):
    """Warming-up test."""
    manifest = read_manifest(manifest_path)
    rng = random.Random(random_seed)
    samples = rng.sample(manifest, num_test_cases)
    for idx, sample in enumerate(samples):
        print("Warm-up Test Case %d: %s", idx, sample['audio_filepath'])
        start_time = time.time()
        transcript = audio_process_handler(sample['audio_filepath'])
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))


def start_server():
    """Start the ASR server"""
    # prepare data generator
    wav_files, text_labels, _ = process_manifest.get_path_trans()
    words_size, words, word_num_map = process_manifest.create_dict("data/aishell/vocab.txt")
    deepspeech2 = init_model.DeepSpeech2(wav_files, text_labels, words_size, words, word_num_map)
    online_model = deepspeech2.init_online_model()
    # prepare ASR inference handler
    def file_to_transcript(filename):
        stream_test = assert_zero.FileParser()
        read_file = stream_test.read_file(filename)
        if read_file == 1 or stream_test.run(0):
            print "recive a empty wav file"
            return ""
        else:
            result_transcript = deepspeech2.predict(re.split('',filename), ["deepspeech2"])
            return result_transcript

    # warming up with utterrances sampled from Librispeech
    print('-----------------------------------------------------------')
    print('Warming up ...')
    deepspeech2.test()
    print('-----------------------------------------------------------')

    # start the server
    server = AsrTCPServer(
        server_address=(args.host_ip, args.host_port),
        RequestHandlerClass=AsrRequestHandler,
        speech_save_dir=args.speech_save_dir,
        audio_process_handler=file_to_transcript)
    print("ASR Server Started.")
    server.serve_forever()


def main():
    print_arguments(args)
    start_server()


if __name__ == "__main__":
    main()
