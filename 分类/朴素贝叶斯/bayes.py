#coding:utf-8
from numpy import *

# 词表到向量的转换
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  # 1 侮辱性言论，0 非侮辱性言论
    return postingList,classVec # 进行词条切分后的文档集合，无标点； 类别标签集合
                 
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空set
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 创建两个集合的并集
    return list(vocabSet) #在所有文档中出现的不重复的词列表

# 得到文档的词汇向量
def setOfWords2Vec(vocabList, inputSet):
    # 词汇表vocabList中的单词在文档inputSet中是否出现，
    # 1出现，0未出现,这样转换为词向量returnVec
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # 词袋模型是 +1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#朴素贝叶斯分类器训练函数，得到两个类别的概率向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) # 文档数
    numWords = len(trainMatrix[0]) # 单词数
    # (标签向量中1的和，即侮辱性的个数）/ 总文档数目 ，类别侮辱出现的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = ones(numWords); p1Num = ones(numWords)
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    # print p0Num
    # print p1Num
    # p0Denom = 2.0; p1Denom = 2.0 #总词数
    p0Denom = 0.0; p1Denom = 0.0 #总词数

    for i in range(numTrainDocs):
        if trainCategory[i] == 1: # 如果这篇文档是侮辱性的，则词对应的个数加1，该文档的总次数加1
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = log(p1Num/p1Denom)
    # p0Vect = log(p0Num/p0Denom)
    p1Vect = (p1Num/p1Denom)
    p0Vect = (p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
#词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 便利函数，（封装一些操作）测试函数
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'分类: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'分类: ',classifyNB(thisDoc,p0V,p1V,pAb)

# 切分文本,文件解析为单词列表
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

  # 垃圾邮件测试
def spamTest():
    docList=[]; classList = []; fullText =[]
    #读取文件，并解析为字符串列表
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)  #列表中全是解析的单词列表，每一项都是列表
        fullText.extend(wordList) # 列表中全是单词，每一项都是单词
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #print docList
    vocabList = createVocabList(docList) #得到全部单词，不重复，每一项都是单词
    # 随机选择测试集，训练集下标
    trainingSet = range(50); testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #print trainingSet,testSet
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)