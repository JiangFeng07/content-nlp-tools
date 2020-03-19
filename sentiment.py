#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel


import tensorflow as tf
import pandas as pd
from tensorflow.contrib import lookup

from data_utils.tokenization import load_vocab_ids, text_to_sequence
from tensorflow_model.text_classification_model import LstmModel


def text_to_ids(text, words):
    ids = text_to_sequence(text, words)
    word_ids = tf.keras.preprocessing.sequence.pad_sequences([ids], value=words['[PAD]'], padding='post',
                                                             maxlen=50)
    return word_ids[0]


def parse_line(line, words):
    def get_content(record):
        fields = record.strip().split('\t')
        if len(fields) != 2:
            raise ValueError("invalid record %s" % record)
        text = text_to_ids(fields[0], words)
        if fields[1] == '0':
            label = [1, 0]
        else:
            label = [0, 1]
        return [text, label]

    result = tf.py_func(get_content, [line], [tf.int32, tf.int64])

    # result[0].set_shape([200])
    # result[1].set_shape([2])
    return {"text": result[0], "label": result[1]}


def train():
    train_path = '/Users/lionel/Desktop/train.csv'
    valid_path = '/Users/lionel/Desktop/valid.csv'

    words_path = '/Users/lionel/Desktop/data/review_relation/bert_words.csv'

    words = load_vocab_ids(words_path, sep='\t')

    train_text = []
    train_label = []

    with tf.gfile.GFile(train_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            train_text.append(fields[0])
            train_label.append(int(fields[1]))

    valid_text = []
    valid_label = []

    with tf.gfile.GFile(valid_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            valid_text.append(fields[0])
            valid_label.append(int(fields[1]))

    train_data = [text_to_ids(ele, words) for ele in train_text]
    train_label = tf.keras.utils.to_categorical(train_label, 2)

    valid_data = [text_to_ids(ele, words) for ele in valid_text]
    valid_label = tf.keras.utils.to_categorical(valid_label, 2)

    import numpy as np
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    valid_data = np.array(valid_data)
    valid_label = np.array(valid_label)

    print train_data.shape

    config = {
        'max_sequence_length': 50,
        'vocab_size': 25000,
        'embedding_size': 200,
        'hidden_size': 100,
        'drop_out': 0.2,
        'num_classes': 2,
        'epoch': 10,
        'batch_size': 100,
        'model_path': '/tmp/100'
    }

    model = LstmModel(config).create_model()
    model.summary()

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])

    model.fit(x=train_data,
              y=train_label,
              epochs=config['epoch'],
              batch_size=config['batch_size'],
              validation_data=(valid_data, valid_label),
              verbose=1)
    model.save(config['model_path'])


def predict(text):
    words_path = '/Users/lionel/Desktop/data/review_relation/bert_words.csv'

    words = load_vocab_ids(words_path, sep='\t')
    wordIds = text_to_ids(text, words)
    signature_key = "xiaoxiang"
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                    "/tmp/saved_model/1")
        signature = meta_graph_def.signature_def
        text = signature[signature_key].inputs["text"].name
        outputs = signature[signature_key].outputs["label"].name
        text = sess.graph.get_tensor_by_name(text)
        outputs = sess.graph.get_tensor_by_name(outputs)

        print(sess.run([outputs], feed_dict={text: [wordIds]}))


class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        if numbers is None or len(numbers) <= 0:
            return False
        for i in range(len(numbers)):

            if numbers[i] < 0 or numbers[i] > max(numbers):
                return False
            if i != numbers[i]:
                if numbers[i] == numbers[numbers[i]]:
                    duplication[0] = numbers[i]
                    return True
                else:
                    tmp = numbers[i]
                    numbers[i] = numbers[tmp]
                    numbers[tmp] = tmp
        return False


if __name__ == '__main__':
    solution = Solution()
    res = solution.duplicate([2, 1, 3, 1, 4], [0])
    print res
    # train()
    # text = '提子很新鲜'
    # predict(text)
    # text = '送的太慢了'
    # predict(text)
    #
    # text = '快递小哥送的很快'
    # predict(text)


    # words_path = '/Users/lionel/Desktop/data/review_relation/bert_words.csv'
    #
    # words = load_vocab_ids(words_path, sep='\t')
    #
    # print text_to_id("我喜欢西红柿炒鸡蛋", words)
