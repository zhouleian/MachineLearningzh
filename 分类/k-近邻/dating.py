# coding:utf-8
from numpy import *
import operator
from os import listdir


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# group,labels = createDataSet()
# print group
# 对文本文件处理，分离特征列，目标变量列
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 行数
    # print numberOfLines
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# filename = "datingTestSet2.txt"
# returnMat,classLabelVector = file2matrix(filename)
# print returnMat
# print classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  # 测试集10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化特征值
    m = normMat.shape[0]
    # print m     #1000
    numTestVecs = int(m * hoRatio)
    print numTestVecs  # 100
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对文本文件中的数据，前100行测试，后面的所有训练，其实就是用来计算距离，
        # 参数：用于分类的输入向量，输入的训练样本集，标签向量，用于选择最近邻居的数目
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        #print normMat[i,:]
        print "分类器返回类型: %d, 真实类型: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "错误率: %f" % (errorCount / float(numTestVecs))
    print "错误数量：", errorCount

#预测函数，
def classifyPerson():
    resultList = ['不喜欢', '魅力一般', '极具魅力']
    percentTats = float(raw_input("玩视频游戏所耗时间百分比?"))
    ffMiles = float(raw_input("每年会的的飞行常客里程数?"))
    iceCream = float(raw_input("每周消耗的冰淇淋公升数?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "你对这个人的印象: ", resultList[classifierResult - 1] #自己测试的时候，确实会出错
    print classifierResult
