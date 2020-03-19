#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import logging
import os

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def train(features, labels, model_path):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1,
                                                        shuffle=True, random_state=0)
    logging.info("Train data size: %d, Test data size: %d" % (len(X_train), len(X_test)))

    model = LogisticRegression(penalty='l1', max_iter=20000)

    ##特征归一化后训练
    # minMax = MinMaxScaler()
    # minMax.fit(X_train)
    # features = minMax.transform(features)
    # train_features = minMax.transform(X_train)
    # valid_features = minMax.fit_transform(X_test)
    # model.fit(train_features, y_train)

    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    prediction = model.predict(X_test)
    print(metrics.classification_report(y_test, prediction))


def score(model, X):
    predictions = model.predict_proba(X)
    indexs = np.argmax(predictions, axis=1)
    scores = []
    for i in xrange(len(indexs)):
        if indexs[i] <= 2:
            scores.append(100.0 / 6 * (indexs[i] + 1 - predictions[i][indexs[i]]))
        else:
            scores.append(100.0 / 6 * (indexs[i] + predictions[i][indexs[i]]))
    return scores


def load_train_data(train_path):
    data = pd.read_csv(train_path, sep='\t', header=None)
    labels = data.values[:, 0].astype('int')
    features = data.values[:, 4:12]
    return labels, features


def predict(model, predict_path):
    data = pd.read_csv(predict_path, sep='\t')
    data.dropna()
    bizids = data.values[:, 1]
    biztypes = data.values[:, 2]
    features = data.values[:, 3:11]
    contentbodys = data.values[:, 11]
    scores = score(model, features)
    with open('/tmp/11113.csv', 'w') as writer:
        for i in xrange(len(scores)):
            writer.write("%s\t%s\t%s\t%s\n" % (
                bizids[i], biztypes[i], str(scores[i]), contentbodys[i]))


def classify_main():
    train_path = "/Users/lionel/Desktop/data/review_quality/排版分/train.csv"
    predict_path = "/tmp/test.csv"
    model_path = '/tmp/text_type'
    ##加载训练数据
    # labels, features = load_train_data(train_path)

    ## 训练模型
    # train(features, labels, model_path)

    ##加载模型并训练
    model = joblib.load(model_path)
    predict(model, predict_path)


def plot(features, labels):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推

    logging.info("Tsne Begin")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, method='exact')
    low_dim_embs = tsne.fit_transform(features[:2000])
    logging.info("Tsne Done")

    for j in xrange(len(low_dim_embs)):
        plt.plot([low_dim_embs[j:j + 1, 0]], [low_dim_embs[j:j + 1, 1]], mark[labels[j]])
    plt.show()


def cluster_train(features, model_path):
    # min_max = MinMaxScaler()
    # features = min_max.fit_transform(features)

    model = KMeans(n_clusters=5).fit(features)
    joblib.dump(model, model_path)

    # labels = model.labels_  # 获取聚类标签
    # centroids = model.cluster_centers_  # 获取聚类中心
    # inertia = model.inertia_  # 获取聚类准则的总和

    return model


def cluster_main():
    train_path = "/Users/lionel/Downloads/13新查询-31188191-1558925340104.txt"
    data = pd.read_csv(train_path, sep='\t').dropna()

    reviewids = data.values[:, 0]
    # biztypes = data.values[:, 2]
    contentbodys = data.values[:, 9]

    features = data.values[:, [1, 3, 4, 5, 6, 7, 8]]

    model = cluster_train(features, "/tmp/cluster_type")

    predictions = model.labels_  # 获取聚类标签

    with open("/tmp/cluster_type_predict.csv", 'w') as writer:
        for i in xrange(len(predictions)):
            writer.write("http://www.dianping.com/review/%s\t%s\t%s\n" % (
                reviewids[i], str(predictions[i]), contentbodys[i]))

    plot(features[:1000], predictions)


if __name__ == "__main__":
    cluster_main()
    # out = open('/tmp/cluster_type_train.csv', 'w')
    # with open("/Users/lionel/Downloads/13新查询-30821290-1558690999696.txt", 'r') as reader:
    #     for line in reader:
    #         fields = line.strip().split('\t')
    #
    #         if len(fields) != 12:
    #             continue
    #         out.write(line)
