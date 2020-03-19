#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import argparse

import jieba
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from data_utils.data_processor import OneInputDataProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/tmp/")
FLAGS, _ = parser.parse_known_args()


def seg(texts):
    words = []
    for text in texts:
        word = jieba.cut(text)
        words.append(' '.join(word))
    return words


def tf_idf_feature(examples):
    texts = []
    labels = []
    for example in examples:
        texts.append(example.text_a)
        labels.append(int(example.label))
    words = seg(texts)
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", decode_error='ignore', lowercase=True).fit(words)
    features = vectorizer.fit_transform(words).toarray()

    return features, labels


def main():
    examples = OneInputDataProcessor().get_train_examples(FLAGS.data_dir)
    fit_transform, labels = tf_idf_feature(examples)
    x_train, x_test, y_train, y_test = train_test_split(fit_transform, labels, test_size=0.2,
                                                        shuffle=True, random_state=0)
    # model = LogisticRegression(max_iter=1000)
    model = BernoulliNB()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    print(metrics.classification_report(y_test, predict))


if __name__ == '__main__':
    # main()

    import pandas as pd
    import os

    base_path = ''
    valid_data = pd.read_csv(os.path.join(base_path, 'valid.csv'), sep='[\t|,]', header=None, engine='python')
    # valid_data = valid_data.fillna(value=2.786950)

    train_data = pd.read_csv(os.path.join(base_path, 'train.csv'), sep='[\t|,]', header=None, engine='python')
    # train_data = train_data.fillna(value=2.786950)

    train_labels = train_data.values[:, 0]
    train_features = train_data.values[:, 1:]

    print train_data

    print train_data.min()
    print train_data.max()

    valid_labels = valid_data.values[:, 0]
    valid_features = valid_data.values[:, 1:]

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    minMax = MinMaxScaler()
    minMax.fit(train_features)
    train_features = minMax.transform(train_features)
    valid_features = minMax.fit_transform(valid_features)

    # model = svm.SVC()
    model = LogisticRegression(penalty='l2', max_iter=2000)

    model.fit(train_features, train_labels)

    print model.coef_
    print model.intercept_

    prediction = model.predict(valid_features)

    print metrics.classification_report(valid_labels, prediction)
