#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf

from tensorflow import keras
import os


class RNNLM(object):
    def __init__(self, config):
        self.config = config
        input = keras.Input(shape=(config['sequence_length'],), name='input')
        embedding = keras.layers.Embedding(input_dim=self.config['vocab_size'],
                                           output_dim=self.config['embedding_size'])
        input_embed = embedding(input)
        lstm_layer = keras.layers.LSTM(self.config['hidden_size'], return_sequences=True, dropout=0.2,
                                       recurrent_dropout=0.2)(input_embed)
        gap = tf.keras.layers.GlobalAveragePooling1D()(lstm_layer)

        lstm_drop = keras.layers.Dropout(self.config['keep_drop'])(gap)
        prediction = keras.layers.Dense(self.config['vocab_size'], activation=tf.nn.softmax)(lstm_drop)
        self.model = keras.Model(inputs=input, outputs=prediction)


config = {
    'embedding_size': 200,
    'hidden_size': 200,
    'keep_drop': 0.5,
    'epoch': 1,
    'batch_size': 1000,
    'sequence_len': 4,
    'vocab_size': 21128,
    'sequence_length': 9,
    'file_path': '/tmp/tfrecord'
}


def _parse_text(text):
    features = {
        'reviews': tf.FixedLenFeature([10, ], tf.int64)
    }
    parsed_example = tf.parse_single_example(text, features)
    features = parsed_example['reviews'][0:9]
    target = tf.one_hot(parsed_example['reviews'][9], depth=config['vocab_size'])
    return features, target


texts = []
file_names = []
for subpath in tf.gfile.ListDirectory(config['file_path']):
    if subpath != '_SUCCESS':
        file_names.append(os.path.join(config['file_path'], subpath))
data = tf.data.TFRecordDataset(file_names).map(_parse_text).shuffle(config['batch_size'] * 10).batch(
    config['batch_size'])
iterator = data.make_one_shot_iterator()
next_element = iterator.get_next()

total_count = 0
with tf.Session() as sess:
    while True:
        try:
            total_count += len(sess.run(next_element)[1])
        except tf.errors.OutOfRangeError:
            break
rnn = RNNLM(config)
rnn.model.summary()
rnn.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

rnn.model.fit(data.make_one_shot_iterator(), steps_per_epoch=total_count // config['batch_size'],
              epochs=config['epoch'])
keras.models.save_model(rnn.model, '/tmp/model.h5')


