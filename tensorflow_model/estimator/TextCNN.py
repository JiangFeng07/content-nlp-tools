#!/usr/bin/env python2
# -*- coding:utf-8 -*- 
# Author: lionel
import os

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def text_cnn_model(features, labels, mode, params):
    embeddings = tf.get_variable(name='embedding', dtype=tf.float32,
                                 shape=[params['vocab_size'], params['embedding_size']])
    sentences = tf.nn.embedding_lookup(embeddings, features)
    sentences = tf.expand_dims(sentences, -1)

    pooled_outputs = []
    for filter_size in params['filter_sizes']:
        conv = tf.layers.conv2d(sentences, filters=params['num_filters'],
                                kernel_size=[filter_size, params['embedding_size']], strides=(1, 1), padding='VALID',
                                activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(conv, pool_size=[params['sentence_max_len'] - filter_size + 1, 1],
                                       strides=(1, 1), padding='VALID')
        pooled_outputs.append(pool)

    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, shape=[-1, params['num_filters'] * len(params['filter_sizes'])])

    if mode == tf.estimator.ModeKeys.TRAIN:
        h_pool_flat = tf.layers.dropout(h_pool_flat, params['drop_out'])

    logits = tf.layers.dense(h_pool_flat, params['num_classes'], activation=None)

    predicted_classes = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes_id': predicted_classes[:, tf.newaxis],
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    labels = tf.argmax(labels, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _parse_text(text):
    features = {
        'dish': tf.FixedLenFeature([15, ], tf.int64),
        'label': tf.FixedLenFeature([2, ], tf.int64)
    }
    parsed_example = tf.parse_single_example(text, features)
    text = parsed_example['dish']
    label = parsed_example['label']
    return text, label


def input_fn(file_path, mode='train', batch_size=200, epochs=1):
    files = []
    for file in tf.gfile.ListDirectory(file_path):
        if file == '_SUCCESS':
            continue
        files.append(os.path.join(file_path, file))
    data = tf.data.TFRecordDataset(files).map(_parse_text)
    if mode == tf.estimator.ModeKeys.TRAIN:
        data = data.shuffle(buffer_size=batch_size * 10)
    return data.repeat(epochs).batch(batch_size).prefetch(1)


if __name__ == "__main__":
    train_files = "/Users/lionel/Desktop/data/dish/dish_verify/data/train_tfrecord"
    valid_files = "/Users/lionel/Desktop/data/dish/dish_verify/data/valid_tfrecord"

    run_config = tf.estimator.RunConfig(model_dir='/tmp/model', log_step_count_steps=100)
    classifier = tf.estimator.Estimator(model_fn=text_cnn_model, params={
        'vocab_size': 25000,
        'embedding_size': 50,
        'filter_sizes': [3, 4, 5],
        'num_filters': 100,
        'sentence_max_len': 15,
        'drop_out': 0.2,
        'num_classes': 2
    }, config=run_config)

    classifier.train(input_fn=lambda: input_fn(train_files))
    eval_results = classifier.evaluate(input_fn=lambda: input_fn(valid_files, mode='valid'))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_results))


    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_files))
    # eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(valid_files, mode='valid'), start_delay_secs=20,
    #                                   throttle_secs=100)
    # tf.estimator.train_and_evaluate(estimator=classifier, train_spec=train_spec, eval_spec=eval_spec)
