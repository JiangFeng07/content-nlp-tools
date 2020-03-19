#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

def distance(embedding_a, embedding_b):
    """
    :param embedding_a: 数组a
    :param embedding_b: 数组b
    :return: 欧式距离，cos距离
    """
    import numpy as np
    from scipy.spatial.distance import pdist

    array_a = np.array([float(ele) for ele in embedding_a.split('_')])
    array_b = np.array([float(ele) for ele in embedding_b.split("_")])

    return np.linalg.norm(np.array(array_a) - np.array(array_b)), pdist(np.vstack([array_a, array_b]), 'cosine')[0]
