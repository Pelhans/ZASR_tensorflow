#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

def conv2d(batch_x, filter_shape, strides, pool_size, hyparam, use_dropout=False):
    filter = tf.get_variable("filter",
                            shape=filter_shape,
                            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                            initializer=tf.truncated_normal_initializer(stddev=0.01),
                            dtype=tf.float32)
    conv = tf.nn.conv2d(batch_x, filter, strides, padding='SAME' )
    conv = tf.nn.relu(conv)
    
    conv = tf.nn.max_pool(conv,
                         ksize=[1, 1, pool_size, 1],
                         strides=[1, 1, pool_size, 1],
                         padding='SAME')
    if use_dropout:
        conv = tf.nn.dropout(conv, 0.5)

    return conv

def lookahead_cnn(inputs, filter_shape, pool_size, seq_length, hyparam, use_dropout=True):
    # combine 2 ahead inputs

    lcnn_inputs = []
    h = tf.get_variable('h', shape=[3, hyparam.n_cell_dim], 
                        initializer=tf.random_normal_initializer(stddev=hyparam.h_stddev))
    b = tf.get_variable('b', shape=[hyparam.n_cell_dim], 
                       initializer=tf.random_normal_initializer(stddev=hyparam.b_stddev))
    print "seq_length: ", seq_length
   
    if np.shape(inputs)[1] < 3:
        print "To short to user lookahead_cnn, the inputs should larger than 2"
        return inputs
    else:
        #  range only receive a interger, so I don't know how to iter it to add the outputs of time step t, t+1,t+2
        for j in range(hyparam.batch_size):
            lcnn_inputs = [[inputs[j][i], inputs[j][i+1], inputs[j][i+2]] for i in range(np.shape(inputs)[1]-2) ]
            lcnn_inputs.append([inputs[j][-2], inputs[j][-1], np.zeros(np.shape(inputs[j][-1])).tolist()])
            lcnn_inputs.append([inputs[j][-1]. np.zeros(np.shape(inputs[j][-1])).tolist(), np.zeros(np.shape(inputs[j][-1])).tolist()])
    
    lcnn_inputs = tf.add( tf.matmul(lcnn_inputs, h), b)
    
    # need reshape inputs with [batch_size, height, withdth, 1]

    lcnn_layer = conv2d(lcnn_inputs, filter_shape, pool_size, use_dropout)

    return lcnn_layer

def BiRNN(inputs, seq_length, hyparam, use_dropout=False):

    # forward
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hyparam.n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=hyparam.keep_dropout_rate)

    # backward
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hyparam.n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=hyparam.keep_dropout_rate)

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=inputs,
                                                             dtype=tf.float32,
                                                             time_major=True, # if time_major is True, the input shape should be [time_steps, batch_size, ...]
                                                             sequence_length=seq_length)
    


    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, hyparam.batch_size, 2 * hyparam.n_cell_dim])

    if use_dropout :
        outputs = tf.nn.dropout(outputs, 0.5)
    
    return outputs
