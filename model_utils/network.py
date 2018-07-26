#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

def conv2d(batch_x, filter_shape, pool_size, hyparam, use_dropout=False):
    filter = tf.get_variable("filter",
                            shape=filter_shape,
                            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            dtype=tf.float32)
    conv = tf.nn.conv2d(batch_x, filter, [1, 1, hyparam.n_input + 2*hyparam.n_input*hyparam.n_context, 1], padding='VALID' )
    conv = tf.nn.relu(conv)
    pool = tf.nn.max_pool(conv,
                         ksize=[1, 1, pool_size, 1],
                         strides=[1, 1, pool_size, 1],
                         padding='SAME')

    if use_dropout:
        pool = tf.nn.dropout(pool, 0.5)


    print "np.shape(pool): ", np.shape(pool)

    return pool

def lookahead_cnn(inputs, filter_shape, pool_size, hyparam, use_dropout=True):
    # combine 2 ahead inputs

    lcnn_inputs = []
    h = tf.get_variable('h', shape=[3, hyparam.n_cell_dim], 
                        initializer=tf.random_normal_initializer(stddev=hyparam.h_stddev))
    b = tf.get_variable('b', shape=[hyparam.n_cell_dim], 
                       initializer=tf.random_normal_initializer(stddev=hyparam.b_stddev))

    if len(inputs) < 3:
        print "To short to user lookahead_cnn, the inputs should larger than 2"
        return inputs
    else:
        lcnn_inputs = [[inputs[i], inpust[i+1], inputs[i+2]] for i in range(np.shape(inputs)[0]-2) ]
        lcnn_inputs.append([inputs[-2], inputs[-1], np.zeros(np.shape(inputs[-1])).tolist()])
        lcnn_inputs.append([inputs[-1]. np.zeros(np.shape(inputs[-1])).tolist(), np.zeros(np.shape(inputs[-1])).tolist()])
    
    lcnn_inputs = tf.add( tf.matmul(lcnn_inputs, h), b)
    
    # need reshape inputs with [batch_size, height, withdth, 1]

    lcnn_layer = conv2d(lcnn_inputs, filter_shape, pool_size, use_dropout)

    return lcnn_layer

def BiRNN(inputs, seq_length, hyparam):
    # forward
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hyparam.n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=hyparam.keep_dropout_rate)

    # backward
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hyparam.n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=hyparam.keep_dropout_rate)

#    birnn = tf.reshape(layer_3, [-1, batch_x_shape[0], self.hyparam.n_hidden_3])

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=inputs,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)

    outputs = tf.concat(outputs, 2)
    # shape [n_steps*self.hyparam.batch_size, 2*self.hyparam.n_cell_dim]
    outputs = tf.reshape(outputs, [-1, 2 * hyparam.n_cell_dim])

    if use_dropout :
        outputs = tf.nn.dropout(outputs, 0.5)
    
    return outputs
