#!/usr/bin/env python2
# -*- coding:utf-8 -*- 
# Author: lionel
import warnings

import tensorflow as tf

import os
import pandas as pd
import numpy as np
import time

from data_utils.config import BaseConfig
from data_utils.tokenization import load_vocab_ids, text_to_sequence
from tensorflow_model.text_classification_model import TextCNNModel, BilstmModel

warnings.simplefilter('ignore')
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_size", 10000, "train data shuffle buffer size")
flags.DEFINE_integer("batch_size", 1000, "train data batch size")
flags.DEFINE_integer("epoch", 5, "epoch size")
flags.DEFINE_string("train_path", "../data/dish/dish_verify/data/train_tfrecord/",
                    "The path of train data")
flags.DEFINE_string("valid_path", "../data/dish/dish_verify/data/valid_tfrecord/",
                    "The path of train data")
flags.DEFINE_string("predict_path", "/tmp/2.csv",
                    'The path of predict data')
flags.DEFINE_string("word_path", "../chinese_L-12_H-768_A-12/vocab.txt",
                    'The path of words dictionary')
flags.DEFINE_string("model_path", "/tmp/model2",
                    "The path of saved model")
flags.DEFINE_string("model_config_path", "../data/dish/dish_verify/config.json",
                    "The path of model config path")

def _parse_text2(text):
    features = {
        'dish': tf.FixedLenFeature([10, ], tf.int64),
        'dishname': tf.FixedLenFeature([10, ], tf.int64),
        'label': tf.FixedLenFeature([2, ], tf.int64)
    }
    parsed_example = tf.parse_single_example(text, features)
    text_a = parsed_example['dish']
    text_b = parsed_example['dishname']
    label = parsed_example['label']
    return text_a, text_b, label


def _parse_text(text):
    features = {
        'dish': tf.FixedLenFeature([15, ], tf.int64),
        'label': tf.FixedLenFeature([2, ], tf.int64)
    }
    parsed_example = tf.parse_single_example(text, features)
    text = parsed_example['dish']
    label = parsed_example['label']
    return text, label


def _parse_text3(text):
    features = {
        'reviews': tf.FixedLenFeature([10, ], tf.int64)
    }
    parsed_example = tf.parse_single_example(text, features)
    review = parsed_example['reviews'][0:4]
    target = tf.one_hot(parsed_example['reviews'][4], depth=21128)
    return review, target


def input_fn(filenames, num_epochs=None, shuffle=True, batch_size=200):
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_text)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def predict_input_fn(text_id, label):
    dataset = tf.data.Dataset.from_tensor_slices((text_id, label))
    return dataset.batch(100)


def train():
    train_file_names = []
    for subpath in tf.gfile.ListDirectory(FLAGS.train_path):
        if subpath != "_SUCCESS":
            train_file_names.append(os.path.join(FLAGS.train_path, subpath))
    valid_file_names = []
    for subpath in tf.gfile.ListDirectory(FLAGS.valid_path):
        if subpath != "_SUCCESS":
            valid_file_names.append(os.path.join(FLAGS.valid_path, subpath))

    # dataset = input_fn(valid_file_names, num_epochs=4, batch_size=100)
    # iterator = dataset.make_one_shot_iterator()
    # next = iterator.get_next()
    # with tf.Session() as sess:
    #     print sess.run(next)[0].shape
    #     print sess.run(next)
    #     print sess.run(next)
    config = BaseConfig.from_json_file(FLAGS.model_config_path).to_dict()
    model = TextCNNModel(config)
    model = model.create_model()
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])

    model = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=FLAGS.model_path)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_file_names, num_epochs=2, batch_size=500))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(valid_file_names, shuffle=False),
                                      start_delay_secs=20, throttle_secs=100)

    start = time.time()
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    end = time.time()
    tf.logging.info("Train and valid time is %ds", end - start)


def predict():
    word_index = load_vocab_ids(FLAGS.word_path)

    data = pd.read_csv(FLAGS.predict_path, header=None)
    # shopid = data.values[:, 0]
    # textid = data.values[:, 1]
    texts = data.values[:, 0]

    text_id = [text_to_sequence(text.decode('utf-8'), word_index) for text in texts]

    text_id = tf.keras.preprocessing.sequence.pad_sequences(text_id, value=word_index['pad'], padding='post',
                                                            maxlen=15)
    print text_id[:10]
    label = np.zeros((len(texts), 2)).astype(np.int64)
    # a = serialize_example(text_id[0], label[0])
    # print tf.train.Example.FromString(a)



    # iterator = dataset.make_one_shot_iterator()
    # next = iterator.get_next()
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             x, y = sess.run(next)
    #             print
    #         except tf.errors.OutOfRangeError:
    #             break
    #
    config = BaseConfig.from_json_file(FLAGS.model_config_path).to_dict()
    model = BilstmModel(config)
    model = model.create_model()
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model = tf.keras.estimator.model_to_estimator(keras_model=model)

    # tf.estimator.Estimator.predict()
    inference = model.predict(
        input_fn=lambda: predict_input_fn(text_id, label),
        checkpoint_path=FLAGS.model_path)

    result = [0 if ele['dense'][0] > 0.5 else 1 for ele in inference]
    with tf.gfile.GFile("/tmp/1.csv", 'w') as writer:
        for i in xrange(len(result)):
            writer.write("%s\t%d\n" % (texts[i], result[i]))


if __name__ == "__main__":
    train()
    # # # predict()
    # import tensorflow as tf
    # import os
    #
    # file_path = "/tmp/tfrecord"
    # file_names = []
    # for subpath in tf.gfile.ListDirectory(file_path):
    #     if subpath != '_SUCCESS':
    #         file_names.append(os.path.join(file_path, subpath))
    # # data = tf.data.TFRecordDataset(file_names[:10]).map(_parse_text3).map(
    # #     lambda X, y: (X, tf.keras.utils.to_categorical(y, 21128))).batch(100)
    # data = tf.data.TFRecordDataset(file_names[:10]).map(_parse_text3).batch(100)
    # iterator = data.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # # X, y = next_element[0], tf.one_hot(next_element[1], 21128)
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             print sess.run(next_element[0])
    #             print sess.run(next_element[1])
    #             break
    #         except tf.errors.OutOfRangeError:
    #             break
    # tf.nn.nce_loss()
