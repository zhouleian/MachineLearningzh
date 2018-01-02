#coding:utf-8
import logRegres
from numpy import *
d,la= logRegres.loadDataSet()
# print(d)
# print (la)
# 梯度上升
# weights = logRegres.gradAscent(d,la)
# logRegres.plotBestFit(weights.getA())
#weights = logRegres.stocGradAscent0(array(d),la)

# 随机梯度上升
# weights = logRegres.stocGradAscent1(array(d),la,20)
# logRegres.plotBestFit(weights)
# for i in range(4):
#     print(i)
# t = [[0,1],[1,0],[1,1]]
# t = mat(t)
# print shape(t)
#
# t2 = [0,1]
# print t2
# print mat(t2).transpose()


# 病马预测
# print(logRegres.colicTest())
logRegres.multiTest()
