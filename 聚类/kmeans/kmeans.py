#coding:utf-8

# 聚类任务，无标签分类
from sklearn import cluster, datasets
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
# 返回特征集，标签集，两个特征x,y
def loadDataSet1(filename):
    dataMat = []; labelMat = []
    x = [];y = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 每一项为列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        x.append(float(lineArr[0]))
        y.append(float(lineArr[1]))
        #每一项为标签
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat,x,y
#返回特征集，两个特征x1,x2
def loadDataSet(filename):
    dataMat = []; labelMat = []
    x = [];y = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 每一项为列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        x.append(float(lineArr[0]))
        y.append(float(lineArr[1]))
    return dataMat,x,y

# 通过畸变函数预测一下最好的 k 取值,用testSet.txt数据得到了最好的k是4
# 可以在下面的kmeans_test中测试一下，分别估计k值2,3，4,5,8，的评估结果，看看是不是4最好
def get_k():
    dataSet,x1, x2 = loadDataSet('testSet.txt')
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(
            X, kmeans.cluster_centers_, "euclidean"), axis=1)) / X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    # plt.ylabel(u'平均畸变程度')
    plt.ylabel('jibian')
    # plt.title(u'用肘部法则来确定最佳的K值')
    plt.title('best K')
    plt.show()

# k-means聚类
def kmeans_test():
    # 原始数据

    dataSet,x1,x2 = loadDataSet('testSet.txt')
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    plt.figure(figsize=(15, 20))
    p1 = plt.subplot(3, 2, 1)
    p1.scatter(x1, x2, c='b',marker='o')

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    # markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
    tests = [2, 3, 4, 5, 8]
    subplot_counter = 1
    for k in tests:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter) # 占据第subplot_counter个位置
        # kmeans_model = KMeans(n_clusters=k).fit(X) #训练
        # kmeans_model = KMeans(n_clusters=k).fit_predict(X) #训练
        kmeans_model = KMeans(n_clusters=k, random_state=9).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans_model)
        # plt.show()
        plt.title('K = %s, C = %.03f' %
                  (k,metrics.calinski_harabaz_score(X, kmeans_model)))
        # for i, j in enumerate(kmeans_model.labels_):
        #     plt.plot(x1[i], x2[i], color=colors[j],
        #              marker=markers[j], ls='None')
        #     plt.title(u'K = %s, s = %.03f' %
        #               (k, metrics.silhouette_score
        #               (X, kmeans_model.labels_, metric='euclidean')))
    plt.show()


if __name__ == '__main__':
    # get_k()
    kmeans_test()
