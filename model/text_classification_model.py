#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf
from tensorflow import keras


class Model(object):
    def __init__(self, config):
        self.config = config

    def create_model(self):
        raise NotImplementedError


class BilstmModel(Model):
    def create_model(self):
        input_a = keras.Input(shape=(self.config['max_sequence_length'],), name='input_a')
        embedding = keras.layers.Embedding(input_dim=self.config['vocab_size'],
                                           output_dim=self.config['embedding_size'])
        input_embed = embedding(input_a)
        bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(self.config['hidden_size'], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(
            input_embed)
        for i in range(self.config['num_hidden_layers'] - 1):
            bilstm = keras.layers.Bidirectional(
                keras.layers.LSTM(self.config['hidden_size'], return_sequences=True, dropout=0.2,
                                  recurrent_dropout=0.2))(
                bilstm)
        gap = tf.keras.layers.GlobalAveragePooling1D()(bilstm)
        gap_drop = keras.layers.Dropout(self.config['drop_out'])(gap)
        predictions = keras.layers.Dense(self.config['num_classes'], activation=tf.nn.softmax)(gap_drop)
        model = keras.Model(inputs=input_a, outputs=predictions)
        model.summary()

        return model


class TextCNNModel(Model):
    def create_model(self):
        main_input = keras.Input(shape=(self.config['max_sequence_length'],), name='input_a')
        embedding = keras.layers.Embedding(input_dim=self.config['vocab_size'],
                                           output_dim=self.config['embedding_size'],
                                           input_length=self.config['max_sequence_length'])
        embedding = embedding(main_input)
        cnn = []
        for ele in [3, 4, 5]:
            cnn1 = keras.layers.Convolution1D(filters=256, kernel_size=ele, padding='same', strides=1,
                                              activation='relu')(
                embedding)
            cnn1 = keras.layers.MaxPool1D(pool_size=4)(cnn1)
            cnn.append(cnn1)

        cnn = keras.layers.concatenate(cnn, axis=-1)
        flatten = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.2)(flatten)
        main_output = keras.layers.Dense(self.config['num_classes'], activation=tf.nn.softmax)(drop)
        model = keras.Model(inputs=main_input, outputs=main_output)

        model.summary()
        return model
