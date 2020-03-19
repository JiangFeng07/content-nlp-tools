#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf
import re

from data_utils.tokenization import load_vocab_ids
from tensorflow_model.ner_model import ner


def process_data(path, word_index, max_len):
    feature_list = []
    tag_list = []
    with tf.gfile.GFile(path, 'r') as reader:
        reader.readline()
        for line in reader:
            fields = line.strip().split('\t')
            if len(fields) != 3:
                continue
            dishnames = fields[2]
            reviewbodys = fields[1]

            for reviewbody in re.split('[。.!！；;?？]', reviewbodys):
                feature = []
                tag = []
                reviewbody = reviewbody.strip()
                if len(reviewbody) < 0:
                    continue
                tags = []
                for i in range(len(reviewbody)):
                    tags.append('O')
                for dishname in dishnames.split(','):
                    try:
                        for ele in re.finditer(dishname, reviewbody):
                            start = ele.span()[0]
                            end = ele.span()[1]
                            tags[start] = 'B_DISH'
                            start = start + 1
                            while start < end:
                                tags[start] = 'I_DISH'
                                start = start + 1
                    except:
                        continue
                for i in range(len(reviewbody)):
                    if 'B_DISH' not in tags:
                        continue
                    if max_len < len(reviewbody):
                        continue
                    feature.append(
                        (word_index.get(reviewbody[i].encode('utf-8'), word_index.get('[UNK]'.encode('utf-8')))))
                    if tags[i] == 'O':
                        tag.append(0)
                    if tags[i] == 'B_DISH':
                        tag.append(1)
                    if tags[i] == 'I_DISH':
                        tag.append(2)
                if len(tag) > 0:
                    feature_list.append(feature)
                    tag_list.append(tag)

    feature_list = tf.keras.preprocessing.sequence.pad_sequences(feature_list,
                                                                 value=word_index.get('[PAD]'.encode('utf-8')),
                                                                 padding='pre', maxlen=max_len)
    tag_list = tf.keras.preprocessing.sequence.pad_sequences(tag_list, value=word_index.get('[PAD]'.encode('utf-8')),
                                                             padding='pre', maxlen=max_len)

    return feature_list, tag_list


if __name__ == '__main__':

    # #处理训练数据
    words_path = "/Users/lionel/Desktop/data/review_relation/bert_words.csv"
    word_index = load_vocab_ids(words_path, sep='\t')
    path = '/Users/lionel/Downloads/review_dish.csv'

    config = {
        'batch_size': 100,
        'max_length': 100,
        'vocab_size': 21128,
        'embedding_size': 100,
        'units': 100,
        'num_tags': 3
    }

    feature_list, tag_list = process_data(path, word_index, config['max_length'])
    # index = feature_list.shape[0] // config['batch_size'] * config['batch_size']
    #
    #
    # dataset = tf.data.Dataset.from_tensor_slices((feature_list[:index], tag_list[:index])).repeat(5).batch(
    #     batch_size=config['batch_size'])
    #
    # iterator = dataset.make_one_shot_iterator()
    # model = ner(config=config, iterator=iterator)
    # tf.summary.scalar("loss", model.loss)
    # tf.summary.scalar("accuracy", model.accuracy)
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('/tmp/ner/tensorboard')
    #
    # # 配置 Saver
    # saver = tf.train.Saver()
    # next_element = iterator.get_next()
    # i = 1
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     while True:
    #         try:
    #             sess.run(model.train_op)
    #
    #             if i % 100 == 0:
    #                 target, prediction, acc, loss = sess.run(
    #                     [model.target, model.viterbi_sequence, model.accuracy, model.loss])
    #
    #                 print('%d batch: accuracy is %f, loss is %f' % (i, acc, loss))
    #                 print(target[0], prediction[0])
    #                 print(target[5], prediction[5])
    #             i += 1
    #         except tf.errors.OutOfRangeError:
    #             break
    #     saver.save(sess=sess, save_path='/tmp/ner/model/model2')

    review = '我特别喜欢吃这家的红烧牛肉、清蒸鲈鱼'
    feature_list = []
    feature = []
    for ele in review:
        feature.append(
            (word_index.get(ele.encode('utf-8'), word_index.get('[UNK]'.encode('utf-8')))))
    feature_list.append(feature)
    feature_list = tf.keras.preprocessing.sequence.pad_sequences(feature_list,
                                                                 value=word_index.get('[PAD]'.encode('utf-8')),
                                                                 padding='pre', maxlen=config['max_length'])

    print(feature_list)

    dataset = tf.data.Dataset.from_tensor_slices((feature_list,
                                                  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0]])).batch(batch_size=1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    model = ner(config=config, iterator=iterator)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path='/tmp/ner/model/model2')
        while True:
            try:
                result = sess.run(model.viterbi_sequence)
                # print(result[0])
                for i in range(len(result)):
                    print(result[i])
                break
            except tf.errors.OutOfRangeError:
                break
