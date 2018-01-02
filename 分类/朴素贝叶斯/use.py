#coding:utf-8

import bayes

#文档集，标签集
listOPosts,listClasses = bayes.loadDataSet()

#包含所有词的列表
myList = bayes.createVocabList(listOPosts)
#print myList

trainMat = [] #文档矩阵
for i in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myList,i))
p0Vect,p1Vect,pAbusive = bayes.trainNB0(trainMat,listClasses)
print p0Vect
print p1Vect
print pAbusive # 侮辱性文档占总文档数目的概率
#print len(trainMat)


#myVec = bayes.setOfWords2Vec(myList,listOPosts[0]) #得到第0篇文档的向量
#print myVec




# vo = [1,2,3]
# print vo.index(2)
# print vo.index(3)
# print vo.index(1)
