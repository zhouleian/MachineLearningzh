#coding:utf-8

from numpy import *
import time

def loadData():
    train_text = open('xigua.txt','r',encoding ='utf-8')
    train_feartures = []
    train_labels = []
    for line in train_text.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(6):
            #print(currLine[i])
            lineArr.append(currLine[i])
        train_feartures.append(lineArr)
        if currLine[6] == '是':
            train_labels.append(1)
        else:
            train_labels.append(0)
    return train_feartures,train_labels

def getNumWords(dataSet):
    vocabSet = set([])
    for one in dataSet:
        for i in one:
            vocabSet = vocabSet | set(i)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # 词袋模型是 +1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def getMat(train_features):
    myList = getNumWords(train_features)
    trainMat = []  # 文档矩阵
    for i in train_features:
         trainMat.append(setOfWords2Vec(myList,i))
    return trainMat

def trainNB(trainMatrix,trainCategory,vocabNum):
    num_train = len(trainMatrix)
    p1 = (sum(trainCategory)+1)/(float(num_train)+2)
    #print(p1)
    p0Denom = 2.0
    p1Denom = 2.0
    p0Num = ones(vocabNum)
    p1Num = ones(vocabNum)
    for i in range(num_train):
        if(trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, p1
# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
if __name__ == '__main__':
    print('开始读取并处理数据')
    time_1 = time.time()

    train_features,train_labels = loadData()
    vocabSet = getNumWords(train_features);vocabNum = len(vocabSet)
    train_mat = getMat(train_features)

    time_2 = time.time()
    print('读取并处理数据消耗 ', time_2 - time_1, ' 秒', '\n')

    print('开始训练')
    p0Vect,p1Vect,p1 = trainNB(train_mat,train_labels,vocabNum)
    time_3 = time.time()
    print('训练耗费 ', time_3 - time_2, ' 秒', '\n')

    print('开始测试')
    testEntry = ['青', '蜷', '浊','淸','陷','滑']
    thisDoc = array(setOfWords2Vec(vocabSet, testEntry))
    res = classifyNB(thisDoc, p0Vect, p1Vect, p1)
    if res ==1:
        print(testEntry, '分类: 好瓜')
    else:
        print(testEntry, '分类: 坏瓜')


    time_4 = time.time()
    print('测试耗费 ', time_4 - time_3, ' 秒', '\n')
    #
    # # sklearn自带模型评估准确的函数
    # score = accuracy_score(test_labels, test_predict)
    # print("正确率： ", score)
