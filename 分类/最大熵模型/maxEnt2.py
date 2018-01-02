#coding:utf-8
import pandas as pd
import numpy as np

import time
import math
import random

# from numpy import *
from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class MaxEnt(object):
    # X 是训练特征集
    def init_params(self, X, Y):
        self.X_ = X
        self.Y_ = set() # 包含所有标签，函数：cal_Pxy_Px
        self.cal_Pxy_Px(X, Y)

        self.N = len(X)                 # 训练集大小
        self.n = len(self.Pxy)          # 书中(x,y)对数
        self.M = 10000.0                # 书91页那个M，但实际操作中并没有用那个值
        # 可认为是学习速率

        self.build_dict()
        self.cal_EPxy()

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}
        # print((type(self.id2xy)))
        for i, (x, y) in enumerate(self.Pxy):
            self.id2xy[i] = (x, y) # 把键值对交换
            self.xy2id[(x, y)] = i

    def cal_Pxy_Px(self, X, Y):
        #default(int)则创建一个类似dictionary对象，
        # 里面任何的values都是int的实例，而且就算是一个不存在的key,
        # d[key] 也有一个默认值，这个默认值是int()的默认值0
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        for i in range(len(X)):
            x_, y = X[i], Y[i]
            self.Y_.add(y) # 包含所有标签
            #x_是一行特征，
            for x in x_:
                self.Pxy[(x, y)] += 1 #每一个特征值和类别的组合的数目
                self.Px[x] += 1 # x_是一个特征样本， 统计每一个特征值得数目

    def cal_EPxy(self):
        #计算书中82页最下面那个期望
        self.EPxy = defaultdict(float)
        for id in range(self.n):
            (x, y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

    def cal_pyx(self, X, y):
        result = 0.0
        # 对每一个特征行中的每个特征，
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x, y)]
                result += self.w[id] # w是权重，最后得到的reslut是累加和
        return (math.exp(result), y)

    def cal_probality(self, X):
        #计算书85页公式6.22
        Pyxs = [(self.cal_pyx(X, y)) for y in self.Y_] # 得到（exp和，类别y）对
       # print(Pyxs)
        Z = sum([prob for prob, y in Pyxs])
        return [(prob / Z, y) for prob, y in Pyxs]

    def cal_EPx(self):
        #计算书83页最上面那个期望
        self.EPx = [0.0 for i in range(self.n)]

        for i, X in enumerate(self.X_):
            Pyxs = self.cal_probality(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        return (x, y) in self.xy2id

    # met.train(train_features, train_labels)
    # X 是 训练特征， Y 是训练标签
    def train(self, X, Y):
        self.init_params(X, Y)
        self.w = [0.0 for i in range(self.n)] # 全0列表
        # 迭代次数
        max_iteration = 1000
        for times in range(max_iteration):
            print('迭代次数 %d' % times)
            sigmas = []
            self.cal_EPx()

            for i in range(self.n):
                sigma = 1 / self.M * math.log(self.EPxy[i] / self.EPx[i])
                sigmas.append(sigma)
            self.w = [self.w[i] + sigmas[i] for i in range(self.n)]
    # 预测
    def test(self, test_features):
        results = []
        for t in test_features:
            result = self.cal_probality(t)
            #print(result)
            results.append(max(result, key=lambda x: x[0])[1])
            # a按result中的【0】位置值在进行排序，返回【1】的值，
            # 因为result是列表，包含两项，每一项都是元组，元组第一项是概率，第二项是类别
            # 实现：选择概率大的类别为预测类别
        return results

def rebuild_features(features):
    '''
    将原feature的（a0,a1,a2,a3,a4,...）
    变成 (0_a0,1_a1,2_a2,3_a3,4_a4,...)形式
    '''
    new_features = []
    for feature in features:
        new_feature = []
        for i, f in enumerate(feature):
            new_feature.append(str(i) + '_' + str(f))
        new_features.append(new_feature)
    return new_features

def loadData():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    # 得到训练特征集、标签集
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        # 到20
        for i in range(21):
            lineArr.append(float(currLine[i]))
        train_features.append(lineArr)
        train_labels.append(float(currLine[21]))

    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        test_features.append(lineArr)
        test_labels.append(float(currLine[21]))
    return train_features, test_features, train_labels, test_labels
if __name__ == "__main__":

    print('开始读取数据')
    time_1 = time.time()

    # 训练集，测试集
    train_features, test_features, train_labels, test_labels = loadData()

    train_features = rebuild_features(train_features)
    test_features = rebuild_features(test_features)

    time_2 = time.time()
    print('读取数据消耗 ', time_2 - time_1, ' 秒', '\n')

    print('开始训练')
    met = MaxEnt()
    #print(len(train_features),len(train_labels))
    met.train(train_features, train_labels)

    time_3 = time.time()
    print('训练耗费 ', time_3 - time_2, ' 秒', '\n')

    print('开始测试')
    test_predict = met.test(test_features)
    #print(test_predict)
    time_4 = time.time()
    print('测试耗费 ', time_4 - time_3, ' 秒', '\n')

    # sklearn自带模型评估准确的函数
    score = accuracy_score(test_labels, test_predict)
    print("正确率： ", score)

