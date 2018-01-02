#coding:utf-8
import xigua
from numpy import *

d,la = xigua.loadDataSet()
weights = xigua.gradAscent(d,la)
# weights = xigua.stocGradAscent0(array(d),la)
# xigua.plotBestFit(weights)
xigua.plotBestFit(weights.getA())# getA,将weights改为数组，array

# xigua.xiguaTest()
# xigua.plotBestFit(weights)
# xigua.loadDataSet()
