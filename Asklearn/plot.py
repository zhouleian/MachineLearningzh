#coding:utf-8

from sklearn import cluster, datasets
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

def one():
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 2, 3, 4, 5, 6]
    fig = plt.figure()
    plt.scatter(x, y, marker='o')
    plt.show()
    # plt.plot(x,y)
    # plt.show()

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
        #每一项为标签
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat,x,y

def plot_hao():
    dataSet, classLabels,x,y = loadDataSet('xiguahao.txt')
    # print(type(dataSet))
    # data = np.array(dataSet)
    # print(type(data))
    # x = dataSet[0]
    # y = dataSet[1]
    # 产生测试数据
    # x = np.arange(1, 10)
    # print(type(x))
    # y = x
    # print(x)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Lemon plot')
    # 设置X轴标签
    plt.xlabel('Density')
    # 设置Y轴标签
    plt.ylabel('Sugar content')
    # 画散点图
    ax1.scatter(x, y, c='b', marker='o')
    dataSet2, classLabels2,x2,y2 = loadDataSet('xiguahuai.txt')
    ax1.scatter(x2, y2, c='r', marker='o')
    plt.legend('ab')
    # 设置图标
    # 显示所画的图
    plt.show()


if __name__=='__main__':
    plot_hao()
