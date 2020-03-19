#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import os

import tensorflow as tf
import numpy as np
import time

from data_utils.config import BaseConfig
from data_utils.tokenization import text_to_sequence, load_vocab_ids


def parse_line(line, vocab, config):
    def get_content(record):
        fields = record.decode('utf-8').strip().split("\t")
        if len(fields) != 4:
            raise ValueError("invalid record %s" % record)
        text_a = list(fields[1])
        text_b = list(fields[3])

        if len(text_a) > config['max_sequence_length']:
            text_a = text_a[:config['max_sequence_length']]

        if len(text_a) < config['max_sequence_length']:
            while len(text_a) < config['max_sequence_length']:
                text_a.append('pad')

        if len(text_b) > config['max_sequence_length']:
            text_b = text_b[:config['max_sequence_length']]

        if len(text_b) < config['max_sequence_length']:
            while len(text_b) < config['max_sequence_length']:
                text_b.append('pad')
        return [fields[0], text_to_sequence(text_a, vocab), fields[2], text_to_sequence(text_b, vocab), fields[1],
                fields[3]]

    result = tf.py_func(get_content, [line], [tf.string, tf.int64, tf.string, tf.int64, tf.string, tf.string])
    print len(result)
    # result[0].set_shape([])
    result[1].set_shape([config['max_sequence_length']])
    # result[2].set_shape([])
    result[3].set_shape([config['max_sequence_length']])
    return {"shop_id": result[0], "text_a_id": result[1], "dish_id": result[2], "text_b_id": result[3],
            "text_a": result[4], "text_b": result[5]}


def input_fn(path, vocab, config, batch_size):
    dataset = tf.data.TextLineDataset(path)

    dataset = dataset.map(lambda line: parse_line(line, vocab, config), num_parallel_calls=4)

    return dataset.batch(batch_size=batch_size)


if __name__ == "__main__":
    config = BaseConfig.from_json_file("/Users/lionel/Desktop/data/dish/dish_similarity/config.json").to_dict()
    path = "/tmp/valid.csv"
    word_path = "/Users/lionel/Desktop/data/dish/dish_similarity/words.csv"
    word_index = load_vocab_ids(word_path)
    signature_key = 'predict_label'
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
        dataset = input_fn(path, word_index, config, 10)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                    "/Users/lionel/Desktop/data/dish/dish_similarity/saved_model/4")
        signature = meta_graph_def.signature_def
        text_a = signature[signature_key].inputs["text_a"].name
        text_b = signature[signature_key].inputs["text_b"].name
        outputs = signature[signature_key].outputs["outputs"].name
        text_a = sess.graph.get_tensor_by_name(text_a)
        text_b = sess.graph.get_tensor_by_name(text_b)
        outputs = sess.graph.get_tensor_by_name(outputs)
        epoch = 1
        start = time.time()
        while True:
            with tf.gfile.GFile(os.path.join('/tmp/', 'predict/%d.csv' % epoch), 'w') as f:
                try:
                    start_epoch = time.time()
                    a = sess.run(next_element)
                    sess.graph.finalize()
                    shop_id = a['shop_id']
                    dish_id = a['dish_id']
                    example_text_a_id = a['text_a_id']
                    example_text_b_id = a['text_b_id']
                    example_text_a = a['text_a']
                    example_text_b = a['text_b']

                    pre = sess.run(outputs, feed_dict={text_a: example_text_a_id, text_b: example_text_b_id})
                    result = np.argmax(pre, axis=1)
                    for i in xrange(len(result)):
                        f.write('%s\t%s\t%s\t%s\t%d\n' % (
                            shop_id[i], example_text_a[i], dish_id[i], example_text_b[i], result[i]))

                    end_epoch = time.time()
                    tf.logging.info("%d epoch predict time is %d seconds" % (epoch, end_epoch - start_epoch))

                except tf.errors.OutOfRangeError:
                    break
            epoch += 1
        end = time.time()
        print tf.logging.info("Total predict time is %d seconds" % (end - start))
