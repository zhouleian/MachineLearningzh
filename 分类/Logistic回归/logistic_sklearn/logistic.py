#coding:utf-8

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

from sklearn import cluster,metrics
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
def load_data():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = [];testLabels=[]
    # 得到训练特征集、标签集
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        # 到20
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))
    return trainingSet,trainingLabels,testSet,testLabels

def loadDataSet(filename):
    dataMat = [];labels = []
    x = [];y = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 每一项为列表
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        x.append(float(lineArr[0]))
        y.append(float(lineArr[1]))
        labels.append(lineArr[2])
    return dataMat,x,y,labels

def plot_data(x,y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('testSet.txt')
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.scatter(x, y, c='b', marker='o')
    plt.show()


# 画图
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('r', 'b', 'g', 'm', 'c')
    # cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # meshgrid转化为矩阵
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # ravel将多维数组降为一维数组
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # 填充颜色，alpha是颜色的深浅
    plt.contourf(xx1, xx2, Z, alpha=0.2)
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],c=colors[idx],marker=markers[idx], label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='m',marker='*', s=55, label='test set')

def logistic_test():
    dataMat, x1, x2, classlabels = loadDataSet('testSet.txt')
    train_feature,test_feature,train_label,test_label = train_test_split(dataMat, classlabels, test_size=0.3, random_state=0)

    features = np.vstack((train_feature, test_feature))
    labels = np.hstack((train_label, test_label))

    classifier = LogisticRegression(C=1000.0, random_state=0)
    classifier.fit(train_feature, train_label)
    plot_decision_regions(features, labels, classifier=classifier, test_idx=range(60, 70))

    # 评价模型
    scores = cross_val_score(classifier, train_feature, train_label, cv=5,scoring='accuracy')
    print('准确率', np.mean(scores), scores)


    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left') # 显示lable
    plt.show()


if __name__=='__main__':
    # dataMat,x1,x2,labels = loadDataSet('testSet.txt')
    # plot_data(x1,x2)
    logistic_test()
