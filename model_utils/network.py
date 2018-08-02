#!/usr/bin/env python
# coding=utf-8

""" Deep Network wraper for BRNN CNN Lookahead CNN """
import tensorflow as tf
import numpy as np

scale = tf.Variable(tf.ones([1]))
offset = tf.Variable(tf.zeros([1]))
variance_epsilon = 0.001

def conv2d(batch_x, filter_shape, strides, pool_size, hyparam, use_dropout=False):
    ''' Convolution Network wraper for tf.conv2d, contains conv relu act and pooling

    :param batch_x: input tensor with shape [batch_size, time_steps, features_dim, in_channel]
    :type batch_x: tensorflow tensor
    :param filter_shape: a list to control filtr shape, [height, width, in_channel, out_channel]
    :type filter_shape: list
    :param strides: a list to control conv strides, [1, height, width, 1]
    :type strides: list
    :param pool_size: pooling size, default value is 2
    :type pool_size: int
    :param hyparam: hyparam config class
    :type hyparam: class
    :param use_dropout: decide wether use dropout
    :type use_dropout: bool
    return a tensor with shape [batch_size, height, width. out_channel]
    '''
    if hyparam.use_bn:
        batch_mean, batch_var = tf.nn.moments(batch_x, [0, 1, 2])
        batch_x = tf.nn.batch_normalization(batch_x, batch_mean, batch_var, offset, scale, variance_epsilon)
    
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

# Ongoing...
def lookahead_cnn(inputs, filter_shape, pool_size, seq_length, hyparam, use_dropout=True):
    ''' Lookahead CNN combines 2 future inputs.
    
    :param inputs: input tensor with shape [batch_size, time_steps, features_dim, in_channel]
    :type inputs: tensorflow tensor
    :param filter_shape: a list to control filtr shape, [height, width, in_channel, out_channel]
    :type filter_shape: list
    :param strides: a list to control conv strides, [1, height, width, 1]
    :type strides: list
    :param pool_size: pooling size, default value is 2
    :type pool_size: int
    :param hyparam: hyparam config class
    :type hyparam: class
    :param use_dropout: decide wether use dropout
    :type use_dropout: bool
    return a tensor with shape [batch_size, height, width. out_channel]
    '''
    # combine 2 ahead inputs

    lcnn_inputs = []
    h = tf.get_variable('h', shape=[3, hyparam.n_cell_dim], 
                        initializer=tf.random_normal_initializer(stddev=hyparam.h_stddev))
    b = tf.get_variable('b', shape=[hyparam.n_cell_dim], 
                       initializer=tf.random_normal_initializer(stddev=hyparam.b_stddev))
    if np.shape(inputs)[1] < 3:
        print "Too short to use lookahead_cnn, the inputs should larger than 2"
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

def BiRNN(inputs, seq_length, batch_x_shape, hyparam, use_dropout=False):
    '''BiRNN wraper with time major

    :param inputs: input tensor with time_major, [time_steps, batch_size, features_dim]
    :type inputs: tensor
    :param seq_length: length of input audio
    :type seq_length: tensor
    :param batch_x_shape: shape of inputs in dim 0
    :type batch_x_shape: tensor
    :param hyparam: model config in class hyparam
    :type hyparam: class
    :param use_dropout: wether use dropout
    :type use_dropout: bool
    return a tensor with [time_steps, batch_size, features_dim]
    '''
    
    if hyparam.use_bn:
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        batch_x = tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, variance_epsilon)

    # forward
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hyparam.n_cell_brnn, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=hyparam.keep_dropout_rate)

    # backward
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hyparam.n_cell_brnn, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=hyparam.keep_dropout_rate)

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=inputs,
                                                             dtype=tf.float32,
                                                             time_major=True, # if time_major is True, the input shape should be [time_steps, batch_size, ...]
                                                             sequence_length=seq_length)
    


    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1,batch_x_shape[0] , 2 * hyparam.n_cell_brnn])

    if use_dropout :
        outputs = tf.nn.dropout(outputs, 0.5)
    
    return outputs
