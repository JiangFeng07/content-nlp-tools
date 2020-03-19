#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import os
import sys
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("file_path", None, "file path")
flags.DEFINE_string("batch_size", None, "batch size")


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'feature0': bytes_feature(feature0),
        'feature1': bytes_feature(feature1),
        'feature2': int64_feature(feature2)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0, f1, f2):
    tf_string = tf.py_func(serialize_example, (f0, f1, f2), tf.string)
    return tf.reshape(tf_string, ())


feature_description = {
    'feature0': tf.FixedLenFeature([], tf.string, default_value=''),
    'feature1': tf.FixedLenFeature([], tf.string, default_value=''),
    'feature2': tf.FixedLenFeature([], tf.int64, default_value=0)
}


def _parse_function(example_proto):
    return tf.parse_single_example(example_proto, feature_description)


def load_tf_records_file(file_path):
    record_iterator = tf.python_io.tf_record_iterator(path=file_path)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        print(example)


def main(_):
    files = []
    for file in os.listdir(FLAGS.file_path):
        files.append(os.path.join(FLAGS.file_path + file))
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.map(lambda line: (tf.string_split([line], delimiter='\t').values)).batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in xrange(sys.maxint):
            try:
                result = sess.run(next_element)
                feature0 = []
                feature1 = []
                feature2 = []
                for j in xrange(len(result)):
                    feature0.append(result[j][0])
                    feature1.append(result[j][1])
                    feature2.append(int(result[j][2]))

                with tf.python_io.TFRecordWriter("/tmp/text%d.tfrecord" % i) as writer:
                    for m in range(len(feature0)):
                        example = serialize_example(feature0[m], feature1[m], feature2[m])
                        writer.write(example)
            except tf.errors.OutOfRangeError:
                break


if __name__ == "__main__":
    tf.app.run()
