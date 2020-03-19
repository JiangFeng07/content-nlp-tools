#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel


import tensorflow as tf
from tensorflow import keras


class Model(object):
    def __init__(self, config, merge_mode):
        self.config = config
        self.merge_mode = merge_mode

    def create_model(self):
        raise NotImplementedError


class BilstmModel(Model):
    def create_model(self):
        input_a = keras.Input(shape=(self.config['max_sequence_length'],), name='input_a')
        input_b = keras.Input(shape=(self.config['max_sequence_length'],), name='input_b')
        embedding = keras.layers.Embedding(input_dim=self.config['vocab_size'],
                                           output_dim=self.config['embedding_size'])
        embedding_a = embedding(input_a)
        embedding_b = embedding(input_b)
        input = None
        if self.merge_mode == 'concat':
            input = keras.layers.concatenate([embedding_a, embedding_b], axis=1)
        if self.merge_mode == 'multiply':
            input = keras.layers.multiply(inputs=[embedding_a, embedding_b])

        bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(self.config['hidden_size'], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(
            input)
        for i in range(self.config['num_hidden_layers'] - 1):
            bilstm = keras.layers.Bidirectional(
                keras.layers.LSTM(self.config['hidden_size'], return_sequences=True, dropout=0.2,
                                  recurrent_dropout=0.2))(
                bilstm)
        gap = tf.keras.layers.GlobalAveragePooling1D()(bilstm)
        gap_drop = keras.layers.Dropout(self.config['drop_out'])(gap)
        predictions = keras.layers.Dense(self.config['num_classes'], activation=tf.nn.softmax)(gap_drop)
        model = keras.Model(inputs=[input_a, input_b], outputs=predictions)
        model.summary()

        return model
