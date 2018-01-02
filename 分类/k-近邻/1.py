#coding:utf-8
from numpy import *
import operator
import dating
#import matplotlib


#group,lables = kNN.createDataSet()
# print group
# print kNN.classify0([0,0],group,lables,3)

#datingDataSet,datingLabels = kNN.file2matrix("datingTestSet2.txt")
dating.datingClassTest()
dating.classifyPerson()

# group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
# labels = ['A','A','B','B']
#classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
# datingLabels[numTestVecs:m],3)
# print group[0,:]
# print group[2:4,:]
