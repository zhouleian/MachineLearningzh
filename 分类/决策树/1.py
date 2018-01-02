#coding:utf-8
#测试
import trees

dataSet,labels = trees.createDataSet()
#print(dataSet)
#print(labels)

#计算香农熵
#print(trees.calcShannonEnt(dataSet))

# a=[1,2,3]
# print(a[1:])

#best = trees.chooseBestFeatureToSplit(dataSet)
#print(best)

#classlist = [0,1,1,0,0]
#print(trees.majorityCnt(classlist))

print(trees.createTree(dataSet,labels))
