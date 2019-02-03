#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import os
import time
import numpy as np
import tensorflow as tf
import warnings
from sklearn import metrics
from tensorflow import keras
from data_utils.config import BaseConfig
from data_utils.data_processor import OneInputDataProcessor, TwoInputDataProcessor
from data_utils.tokenization import load_vocab_ids, features_labels_digitalize
from model import text_classification_model, text_similarity_model

warnings.simplefilter('ignore')
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", None, "Data path")
flags.DEFINE_string('json_file', None, "File which set model parameters")
flags.DEFINE_string('model_name', None, "Model path")
flags.DEFINE_string('signature_name', 'predict_label', "Signature name")
flags.DEFINE_integer('saved_model_version', 1, "Saved model version")
flags.DEFINE_integer('input_format', 2, "input  format, if value=1，mean one input; if value=2, mean two input")
flags.DEFINE_boolean('do_train', False, "Train or not")
flags.DEFINE_boolean('do_valid', False, "Valid or not")
flags.DEFINE_boolean('do_test', False, "Test or not ")
flags.DEFINE_boolean('do_export', False, "Saved model ")
flags.DEFINE_boolean('do_statistic', False, "print recall、accuracy、f1-score or not ")
flags.DEFINE_list('params', None, 'Params')


def train(config):
    """
    训练模型
    :param config: 超参数配置文件
    :return: 预测结果
    """
    word_index = load_vocab_ids(os.path.join(FLAGS.data_dir, 'words.csv'))
    if not tf.gfile.Exists(os.path.join(FLAGS.data_dir, 'model')):
        tf.gfile.MkDir(tf.gfile.Exists(os.path.join(FLAGS.data_dir, 'model')))
    model_path = os.path.join(FLAGS.data_dir, 'model/%s' % FLAGS.model_name)

    train_examples = None
    valid_examples = None
    test_examples = None

    if FLAGS.do_train and FLAGS.do_valid:
        if FLAGS.input_format == 1:
            train_examples = OneInputDataProcessor().get_train_examples(FLAGS.data_dir)
            valid_examples = OneInputDataProcessor().get_valid_examples(FLAGS.data_dir)

        if FLAGS.input_format == 2:
            train_examples = TwoInputDataProcessor().get_train_examples(FLAGS.data_dir)
            valid_examples = TwoInputDataProcessor().get_valid_examples(FLAGS.data_dir)

        train_text_a, train_text_b, train_label_ids = features_labels_digitalize(train_examples, word_index,
                                                                                 config['max_sequence_length'])
        valid_text_a, valid_text_b, valid_label_ids = features_labels_digitalize(valid_examples, word_index,
                                                                                 config['max_sequence_length'])

        model = None
        if FLAGS.input_format == 2:
            model = text_similarity_model.BilstmModel(config, merge_mode='multiply')
        if FLAGS.input_format == 1:
            model = text_classification_model.BilstmModel(config)
        model = model.create_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']), loss='binary_crossentropy',
                      metrics=['accuracy'])

        start = time.time()
        if train_text_b is not None:
            model.fit(x=[train_text_a, train_text_b],
                      y=train_label_ids,
                      epochs=config['epoch'],
                      batch_size=config['batch_size'],
                      validation_data=([valid_text_a, valid_text_b], valid_label_ids),
                      callbacks=[keras.callbacks.EarlyStopping(patience=2)])
        else:
            model.fit(x=train_text_a,
                      y=train_label_ids,
                      epochs=config['epoch'],
                      batch_size=config['batch_size'],
                      validation_data=(valid_text_a, valid_label_ids),
                      callbacks=[keras.callbacks.EarlyStopping(patience=2)])

        end = time.time()
        tf.logging.info("Train time is %ds", end - start)

        model.save(model_path, overwrite=True)

        if FLAGS.do_test:
            if FLAGS.input_format == 1:
                test_examples = OneInputDataProcessor().get_test_examples(FLAGS.data_dir)
            if FLAGS.input_format == 2:
                test_examples = TwoInputDataProcessor().get_test_examples(FLAGS.data_dir)
            test_text_a, test_text_b, test_label_ids = features_labels_digitalize(test_examples, word_index,
                                                                                  config['max_sequence_length'])
            model = keras.models.load_model(model_path)
            if FLAGS.do_statistic:
                if test_text_b is not None:
                    result = model.predict([test_text_a, test_text_b])
                else:
                    result = model.predict(test_text_a)
                print metrics.classification_report(np.argmax(test_label_ids, axis=1), np.argmax(result, axis=1))
            else:
                count = len(test_text_a)
                n = count // config['batch_size'] + 1
                with tf.gfile.GFile(os.path.join(FLAGS.data_dir, 'predict.csv'), 'w') as f:
                    for i in range(n):
                        if (i + 1) * config['batch_size'] >= count:
                            x_a = test_text_a[i * config['batch_size']:]
                            x_b = test_text_b[i * config['batch_size']:]
                        else:
                            x_a = test_text_a[i * config['batch_size']: (i + 1) * config['batch_size']]
                            x_b = test_text_b[i * config['batch_size']: (i + 1) * config['batch_size']]

                        predictions = model.predict_on_batch([x_a, x_b])
                        result = np.argmax(predictions, axis=1)

                        for j in range(len(result)):
                            index = i * config['batch_size'] + j
                            f.write(
                                '%s\t%s\t%s\n' % (test_examples[index].text_a, test_examples[index].text_b, result[j]))

        if FLAGS.do_export:
            model = keras.models.load_model(model_path)
            features = dict()
            x = model.input
            y = model.output
            args = FLAGS.params
            if isinstance(x, list):
                for i in range(len(x)):
                    features[args[i]] = x[i]
            else:
                features[args[0]] = x

            labels = dict()
            if isinstance(y, list):
                for i in range(len(y)):
                    labels[args[len(features)]] = y[i]
            else:
                labels[args[len(features)]] = y

            sess = tf.keras.backend.get_session()

            prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
                inputs=features, outputs=labels)

            valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
            if not valid_prediction_signature:
                raise ValueError("Error: Prediction signature not valid!")

            saved_model_path = os.path.join(FLAGS.data_dir, 'saved_model/%d' % (FLAGS.saved_model_version))
            if tf.gfile.Exists(saved_model_path):
                tf.gfile.DeleteRecursively(saved_model_path)
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={FLAGS.signature_name: prediction_signature},
                legacy_init_op=legacy_init_op)
            builder.save()


def main(_):
    config = BaseConfig.from_json_file(os.path.join(FLAGS.data_dir, 'config.json')).to_dict()
    train(config)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    tf.app.run()
