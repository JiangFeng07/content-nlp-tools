#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import collections
import tensorflow as tf

import six
import unicodedata


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file (char) into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        for line in reader:
            token = convert_to_unicode(line)
            if not token:
                continue
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def load_vocab_ids(vocab_file):
    """Loads a vocabulary file ( char:id ) into a dictionary."""
    vocab = dict()
    with tf.gfile.GFile(vocab_file, "r") as reader:
        for line in reader:
            token = convert_to_unicode(line)
            fields = token.split(':')
            if len(fields) != 2:
                continue
            if fields[0] is None:
                continue
            vocab[fields[0].strip()] = int(fields[1])
    return vocab


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def features_labels_digitalize(examples, word_index, maxlen):
    text_a_ids = []
    text_b_ids = []
    label_ids = []
    for example in examples:
        ids_a = text_to_sequence(example.text_a, word_index)
        text_a_ids.append(ids_a)
        label_ids.append(convert_to_unicode(example.label))

        if example.text_b is None:
            continue
        ids_b = text_to_sequence(example.text_b, word_index)
        text_b_ids.append(ids_b)

    text_a_ids = tf.keras.preprocessing.sequence.pad_sequences(text_a_ids, value=word_index['pad'], padding='post',
                                                               maxlen=maxlen)
    if len(text_b_ids) <= 0:
        text_b_ids = None
    else:
        text_b_ids = tf.keras.preprocessing.sequence.pad_sequences(text_b_ids, value=word_index['pad'],
                                                                   padding='post',
                                                                   maxlen=maxlen)
    label_ids = tf.keras.utils.to_categorical(label_ids, 2)
    return text_a_ids, text_b_ids, label_ids


def text_to_sequence(text, word_index):
    sequence = []
    for word in list(convert_to_unicode(text)):
        sequence.append(word_index.get(word, word_index.get('unk')))
    return sequence
