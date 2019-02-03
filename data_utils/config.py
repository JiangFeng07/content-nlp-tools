#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import six
import json
import copy
import tensorflow as tf


class BaseConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=100,
                 num_hidden_layers=4,
                 embedding_size=100,
                 max_sequence_length=10,
                 batch_size=512,
                 epoch=5,
                 num_classes=2,
                 attention_size=100):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_classes = num_classes
        self.attention_size = attention_size

    @classmethod
    def from_dict(cls, json_object):
        config = BaseConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
