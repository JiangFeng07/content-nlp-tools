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
from data_utils.tokenization import load_vocab_ids, features_labels_digitalize, text_to_sequence
from tensorflow_model import text_classification_model, text_similarity_model

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
                    for i in xrange(n):
                        if (i + 1) * config['batch_size'] >= count:
                            x_a = test_text_a[i * config['batch_size']:]
                            x_b = test_text_b[i * config['batch_size']:]
                        else:
                            x_a = test_text_a[i * config['batch_size']: (i + 1) * config['batch_size']]
                            x_b = test_text_b[i * config['batch_size']: (i + 1) * config['batch_size']]

                        predictions = model.predict_on_batch([x_a, x_b])
                        result = np.argmax(predictions, axis=1)

                        for j in xrange(len(result)):
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


def model_predict(model_path, texts, predict_result_path, word_index):
    reviewids = []
    shoptypes = []
    reviews = []
    for line in texts:
        fields = line.strip().split('\t')
        if len(fields) != 3:
            continue
        reviewids.append(fields[0])
        shoptypes.append(fields[2])
        reviews.append(fields[1])
    review_ids = [text_to_sequence(review, word_index) for review in reviews]
    review_ids = tf.keras.preprocessing.sequence.pad_sequences(review_ids, value=word_index['[PAD]'], padding='post',
                                                               maxlen=500)

    model = tf.keras.models.load_model(model_path)
    result = model.predict(review_ids)
    label = np.argmax(result, axis=1)

    with tf.gfile.GFile(predict_result_path, 'w') as writer:
        writer.write(
            "reviewid\tshoptype\tlabel\tresult\treview\n")
        for i in range(len(reviews)):
            writer.write("%s\t%s\t%s\t%s\t%s\n" % (
                reviewids[i], shoptypes[i], str(label[i]), str(result[i]), reviews[i]))


def model_predict2(model_path, texts, word_index):
    reviewids = []
    reviews = []
    for line in texts:
        fields = line.strip().split('\t')
        if len(fields) != 2:
            continue
        reviewids.append(fields[0])
        reviews.append(fields[1])
    review_ids = [text_to_sequence(review, word_index) for review in reviews]
    review_ids = tf.keras.preprocessing.sequence.pad_sequences(review_ids, value=word_index['[PAD]'], padding='post',
                                                               maxlen=500)

    model = tf.keras.models.load_model(model_path)
    result = model.predict(review_ids)
    label = np.argmax(result, axis=1)
    return label


def predict2():
    predict_base_path = "../data/review_relation/predict/content_relation.csv"
    reviews = []
    contentids = []
    reviewlist = []
    with tf.gfile.GFile(predict_base_path, 'r') as reader:
        for line in reader:
            fields = line.strip().split("\t")
            if len(fields) != 2:
                continue
            contentids.append(fields[0])
            reviewlist.append(fields[1])
            reviews.append(line.strip())

    words_path = "../data/review_relation/bert_words.csv"
    word_index = load_vocab_ids(words_path, sep='\t')
    model_base_path = "../data/review_relation/version2/"
    predict_result_base_path = "../data/review_relation/content_pic_relation/content_relation_res.csv"
    food_label = model_predict2(os.path.join(model_base_path, 'food2.h5'), reviews, word_index)
    jiudian_label = model_predict2(os.path.join(model_base_path, 'jiudian.h5'), reviews, word_index)
    liren_label = model_predict2(os.path.join(model_base_path, 'liren.h5'), reviews, word_index)
    yule_label = model_predict2(os.path.join(model_base_path, 'yule.h5'), reviews, word_index)
    gouwu_label = model_predict2(os.path.join(model_base_path, 'gouwu.h5'), reviews, word_index)

    with tf.gfile.GFile(predict_result_base_path, 'w') as writer:
        writer.write(
            "contentid\tfood\tjiudian\tliren\tyule\tgouwu\tcontentbody\n")
        for i in xrange(len(contentids)):
            writer.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                contentids[i], str(food_label[i]), str(jiudian_label[i]), str(liren_label[i]), str(yule_label[i]),
                str(gouwu_label[i]), reviews[i]))


def predict():

    predict_base_path = "../data/review_relation/predict/content_relation.csv"

    food_reviews = []
    liren_reviews = []
    jiudian_reviews = []
    yule_reviews = []
    gouwu_reviews = []

    with tf.gfile.GFile(predict_base_path, 'r') as  reader:
        for line in reader:
            fields = line.strip().split('\t')
            if fields[2] == "美食":
                food_reviews.append(line)
            elif fields[2] == "丽人":
                liren_reviews.append(line)
            elif fields[2] == "酒店":
                jiudian_reviews.append(line)
            elif fields[2] == "休娱":
                yule_reviews.append(line)
            else:
                gouwu_reviews.append(line)

    words_path = "../data/review_relation/bert_words.csv"
    word_index = load_vocab_ids(words_path, sep='\t')

    model_base_path = "../data/review_relation/version2/"
    predict_result_base_path = "../data/review_relation/predict_result/"
    model_predict(os.path.join(model_base_path, 'food2.h5'), food_reviews,
                  os.path.join(predict_result_base_path, 'food_predict_result2.csv'), word_index)
    model_predict(os.path.join(model_base_path, 'jiudian.h5'), jiudian_reviews,
                  os.path.join(predict_result_base_path, 'jiudian_predict_result2.csv'), word_index)
    model_predict(os.path.join(model_base_path, 'liren.h5'), liren_reviews,
                  os.path.join(predict_result_base_path, 'liren_predict_result2.csv'), word_index)
    model_predict(os.path.join(model_base_path, 'yule.h5'), yule_reviews,
                  os.path.join(predict_result_base_path, 'yule_predict_result2.csv'), word_index)
    model_predict(os.path.join(model_base_path, 'gouwu.h5'), gouwu_reviews,
                  os.path.join(predict_result_base_path, 'gouwu_predict_result2.csv'), word_index)


def main(_):
    config = BaseConfig.from_json_file(os.path.join(FLAGS.data_dir, 'config.json')).to_dict()
    train(config)


if __name__ == '__main__':
    # predict()
    # predict2()
    # tf.app.run()
    # sess = tf.Session()
    # a = tf.one_hot([0, 1, 2, 3, 4, 5, 6, 7], depth=8)
    # print sess.run(a[1])
    #
    #
    # from tensorflow import  keras
    #
    # print keras.utils.to_categorical()

    path = '/tmp/part-r-00031'
    dataset = tf.data.TFRecordDataset(path).batch(10)
    iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                x,y = sess.run(iterator.get_next())
                for ele in x:
                    print(ele)
                break
            except tf.errors.OutOfRangeError:
                break
