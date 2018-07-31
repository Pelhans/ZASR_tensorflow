#!/usr/bin/env python
# coding=utf-8


class Config():
    def __init__(self):

        self.b_stddev = 0.046875
        self.h_stddev = 0.046875

        self.n_hidden = 512
        self.n_hidden_1 = 512
        self.n_hidden_2 = 512
        self.n_hidden_5 = 512
        self.n_cell_dim = 512
        self.n_hidden_3 = 2 * 512

        self.learning_rate = 0.001
        self.keep_dropout_rate = 0.95
        self.keep_dropout_rate = 0.95
        self.relu_clip = 20

        self.n_input = 39  # 计算MFCC的个数
        self.n_context = 2  # 对于每个时间点，要包含上下文样本的个数
        self.specgram_type = 'linear'
        self.batch_size = 8 
