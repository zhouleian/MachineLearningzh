#coding:utf-8

from numpy import *
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#阈值比较实现分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))# 创建“行”大小的矩阵
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 得到具有最小错误率的单层决策树，还有返回的最小错误率和估计的类别向量
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T #转置
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf # 无穷大
    # 对数据每个特征，n是数据集的列数，即特征数
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        # print rangeMin,rangeMax
        stepSize = (rangeMax-rangeMin)/numSteps #步长
        # print stepSize
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                #errArr初始化为1的矩阵，如果预测的predictedVals不等于labelMat，那么相应位置为1
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr # 分类误差率
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):# numIt相当于多少个弱分类器
    weakClassArr = []
    m = shape(dataArr)[0] #行数

    D = mat(ones((m,1))/m) # 初始化权重，1/m
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print i,"D:",D.T

        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))  # 计算alpha权重，强分类器的权重。
        print "alpha: ",alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ",classEst.T # 第i个弱分类器得到的预测结果，根据这个结果计算误差等

        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        # print "expon:",expon
        #重新计算权重
        D = multiply(D,exp(expon))
        D = D/D.sum()
        print "D: " ,D
        aggClassEst += alpha*classEst
        print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "错误率: ",errorRate
        print
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst


def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]#行
    aggClassEst = mat(zeros((m,1)))
    # 计算每一个弱分类器
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)



if __name__=='__main__':

    dataMat,classLabels = loadDataSet('horseColicTraining2.txt')
    classifier,aggC = adaBoostTrainDS(dataMat,classLabels,10)

    testMat,testLabels = loadDataSet('horseColicTest2.txt')
    preclassifier = adaClassify(testMat,classifier)
    err = mat(ones((67,1)))
    print err[preclassifier !=mat(testLabels).T].sum()

    # D = mat(ones((5,1))/5)
    # datMat,classLabels = loadSimpData()
    # print shape(datMat)[0]
    # retArray = ones((shape(datMat)[0], 1))
    # print shape(datMat)[0] # 5
    # print retArray
    # bestStump,minError,bestClasEst = buildStump(datMat,classLabels,D)
    # weakClassArr,aggClassEst= adaBoostTrainDS(datMat,classLabels,9)
    # print weakClassArr
    # print aggClassEst
    # print
    # print adaClassify([0,0],weakClassArr)
    # print bestStump
    # print minError
    # print bestClasEst



