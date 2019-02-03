#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('do_save', False, 'saved model')
flags.DEFINE_boolean('do_test', False, 'test saved model')
flags.DEFINE_string('signature_name', 'predict_label', 'define signature name')
flags.DEFINE_string('model_path', None, 'Tensorflow model path')
flags.DEFINE_string('saved_model_path', None, 'Savel model path ')
flags.DEFINE_list('params', None, 'Params')


def test_text_similarity_model():
    signature_key = FLAGS.signature_name
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    text_1 = np.array([[6538, 3545, 6837, 8093, 296, 1, 0, 0, 0, 0]])
    text_2 = np.array([[7476, 3545, 6837, 8093, 296, 0, 0, 0, 0, 0]])
    with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                    FLAGS.saved_model_path)
        signature = meta_graph_def.signature_def
        text_a = signature[signature_key].inputs["text_a"].name
        text_b = signature[signature_key].inputs["text_b"].name
        outputs = signature[signature_key].outputs["outputs"].name
        text_a = sess.graph.get_tensor_by_name(text_a)
        text_b = sess.graph.get_tensor_by_name(text_b)
        outputs = sess.graph.get_tensor_by_name(outputs)

        print(sess.run([outputs], feed_dict={text_a: text_1, text_b: text_2}))


def save_text_similarity_model():
    model = tf.keras.models.load_model(filepath=FLAGS.model_path)
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
    if (valid_prediction_signature == False):
        raise ValueError("Error: Prediction signature not valid!")

    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.saved_model_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={FLAGS.signature_name: prediction_signature},
        legacy_init_op=legacy_init_op)
    builder.save()


def main(_):
    save_text_similarity_model()


if __name__ == "__main__":
    flags.mark_flag_as_required("model_path")
    flags.mark_flag_as_required("saved_model_path")
    tf.app.run()
