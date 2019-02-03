#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import csv
import os
import tensorflow as tf

from data_utils.tokenization import convert_to_unicode


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_predict_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class OneInputDataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.csv')), set_type='train')

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'valid.csv')), set_type='valid')

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'test.csv')), set_type='test')

    def get_predict_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'predict.csv')), set_type='predict')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            text_a = convert_to_unicode(line[0])
            if set_type == 'predict':
                label = '0'
            else:
                label = convert_to_unicode(line[-1])
            examples.append(InputExample(text_a=text_a, text_b=None, label=label))
        return examples


class TwoInputDataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.csv')), set_type='train')

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'valid.csv')), set_type='valid')

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'test.csv')), set_type='test')

    def get_predict_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'predict.csv')), set_type='predict')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            text_a = convert_to_unicode(line[0])
            text_b = convert_to_unicode(line[1])
            if set_type == 'predict':
                label = '0'
            else:
                label = convert_to_unicode(line[-1])
            examples.append(InputExample(text_a=text_a, text_b=text_b, label=label))
        return examples
