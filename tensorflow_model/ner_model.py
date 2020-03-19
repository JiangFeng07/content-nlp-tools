#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np


class ner(object):
    def __init__(self, config, iterator):
        self.config = config
        self.input, self.target = iterator.get_next()

        self.seq_lengths = np.full(self.config['batch_size'], self.config['max_length'])

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable(name='embedding',
                                        shape=[self.config['vocab_size'], self.config['embedding_size']],
                                        dtype=tf.float32,
                                        initializer=xavier_initializer())
            embed = tf.nn.embedding_lookup(embedding, self.input)

        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config['units'])
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config['units'])

        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embed, dtype=tf.float32)

        # output = tf.concat(outputs, axis=2)

        output_fw, output_bw = outputs
        output = tf.concat([output_fw, output_bw], axis=-1)

        # 在这里设置一个无偏置的线性层
        self.W = tf.get_variable('W', shape=[2 * self.config['units'], self.config['num_tags']], dtype=tf.float32)
        self.b = tf.get_variable('b', shape=[self.config['num_tags']], dtype=tf.float32)
        matricized_output = tf.reshape(output, [-1, 2 * self.config['units']])
        matricized_unary_scores = tf.matmul(matricized_output, self.W) + self.b
        self.scores = tf.reshape(matricized_unary_scores,
                                 [-1, self.config['max_length'], self.config['num_tags']])

        ## softmax
        # self.prediction = tf.cast(tf.argmax(self.scores, axis=-1), tf.int32)
        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.target)
        # mask = tf.sequence_mask(self.seq_lengths)
        # losses = tf.boolean_mask(losses, mask)
        # self.loss = tf.reduce_mean(losses)
        # self.correct_num = tf.cast(tf.equal(self.target, self.prediction), tf.float32)


        ## crf
        # 计算log-likelihood并获得transition_params
        self.log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.target,
                                                                                   self.seq_lengths)
        # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            self.scores, transition_params, self.seq_lengths)

        self.loss = tf.reduce_mean(-self.log_likelihood)

        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        self.correct_num = tf.cast(tf.equal(self.target, self.viterbi_sequence), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_num)
