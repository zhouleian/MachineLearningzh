'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算香农熵,并返回
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {} #字典
    # 统计数据集中的类别和对应的个数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
   # print(labelCounts)
    #for key in labelCounts:
        #print(key,labelCounts[key])
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2

    return shannonEnt

#按照给定特征axis划分数据集，得到的样本数据retDateSet必须是axis列的值为value的
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

    #选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #特征属性数目，因为最后一列是标签列，所以减1
    baseEntropy = calcShannonEnt(dataSet) #计算原始香农熵的值
    bestInfoGain = 0.0; bestFeature = -1

    #遍历所有特征列
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#得到第i列特征值写到featList中
        #print(featList)
        uniqueVals = set(featList) #该列特征值去重值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)#划分数据集
            #计算信息增益
            prob = len(subDataSet)/float(len(dataSet))
            temp = calcShannonEnt(subDataSet)
            #print(i,temp)
            newEntropy += prob * temp

        infoGain = baseEntropy - newEntropy

        #比较得到最好的划分数据集的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#python3里没有iteritems()的函数了，所以，构建一个，以便进行排序
def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

#返回出现次数最多的分类标签，这个实现如果次数一样，返回classList里面前面的那个
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(dict2list(classCount), key=operator.itemgetter(1), reverse=True)

    #这是在python2中可以这么用，python3里没有iteritems()函数了
   # sortedClassCount = sorted(classCount.iteritems(),
       #                       key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #数据集里提取的标签列
    #print(classList,classList[0])
    # 如果所有的类标签相同，直接返回该类标签：统计第一个类标签的数量，如果==所有类标签的个数，
    # 很明显所有类都是第一个类标签的类，直接返回第一个类
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # dataSet[0]是一行数据，如果==1，说明只有一个属性或者用完了所有特征，
    # 直接返回出现次数最多的标签分类
    #print(len((dataSet[0])))
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]#最好的划分特征属性
    #print(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归待用函数createTree()，
    # 得到的返回值将被插入到字典变量myTree中，因此函数终止执行时，字典中将会嵌套很
    # 多代表叶子节点信息的字典数据。
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            

