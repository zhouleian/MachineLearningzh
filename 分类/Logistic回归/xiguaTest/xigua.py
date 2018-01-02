#coding:utf-8

from numpy import *


# 读取文本中的数据，得到特征列表和标签列表
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('xigua3.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
       # print(lineArr)# 每一项为列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #每一项为标签
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升算法，求解最佳回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    # print(dataMatrix.shape)#转换为Numpy矩阵
    # print(dataMatrix)
    labelMat = mat(classLabels).transpose() # 转换为列向量
    m,n = shape(dataMatrix) # m 行 表示m个训练样本 n列，表示n个特征
    alpha = 0.001
    maxCycles = 500 # 迭代次数
    weights = ones((n,1))
    # print(weights)
    # print(dataMatrix*weights)
    # oo=1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)# 预测类别
        # if oo==1 :
        #     print(h.shape)
        #     print(h)
        # oo = 4
        error = (labelMat - h)              # 真实类别和预测类别的差值
        #调整回归系数，梯度下降算法的时候 是 - alpha，这是不同的
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights

# 画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(0.1, 0.9, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.show()

#改进方法，随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    # print(weights.shape)
    # weights2 = ones((n, 1))
    # print(weights)
    # print(weights2)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        # print(h)
        # if i==0:
        #     print(dataMatrix[i])
        #     print(dataMatrix[i].shape)
        #     print((dataMatrix[i] * weights).shape)
        #     print(h)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #每次迭代调整
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选取

            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def xiguaTest():
    xiTrain = open('xigua3.txt')
    xiTest = open('xigua3test.txt')
    trainingSet = []; trainingLabels = []
    # 得到训练特征集、标签集
    for line in xiTrain.readlines():
        currLine = line.strip().split()
        trainingSet.append([float(currLine[0]),float(currLine[1])])
        trainingLabels.append(int(currLine[2]))
    # print(trainingSet)
    # print(trainingLabels)
    # 计算回归系数向量
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 2000)
    # return trainWeights
    errorCount = 0
    numTestVec = 0.0#测试的数目
    for line in xiTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split()

        lineArr =[]
        for i in range(2):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[2]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("错误率: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += xiguaTest()
    print("%d 迭代之后，平均错误率是：%f" % (numTests, errorSum/float(numTests)))
