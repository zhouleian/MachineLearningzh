from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
# print(preprocessing.binarize(X, threshold=2, copy=True)) # 小于threshold的为0
# print(X)

# classif = OneVsRestClassifier(estimator=SVC(random_state=0))
# print(classif.fit(X, y).predict(X)) # output:[0 0 1 1 2]
# print()

# y = LabelBinarizer()
# print(y)
# y = LabelBinarizer().fit_transform(y)
# print(y)
# print()
# print(classif.fit(X, y).predict(X))  # output:[[1 0 0][1 0 0][0 1 0][0 0 0][0 0 0]]
#
# from sklearn.preprocessing import MultiLabelBinarizer
# y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
# y = MultiLabelBinarizer().fit_transform(y)
# print(y)
# print()
# print(classif.fit(X, y).predict(X))
# output:[[1 1 0 0 0][1 0 1 0 0][0 1 0 1 0][1 0 1 0 0][1 0 1 0
from sklearn import datasets

# iris = datasets.load_iris()
# data = iris.data
# print(data.shape)
# 数据集iris的形状(150, 4),iris数据集由150个观察值构成，
# 每个观察值有4个特征，他们的萼片和花瓣的长度和宽度存储在iris.DESCR
# print(iris.DESCR)
# digits = datasets.load_digits()
# print(digits.images.shape)  # (1797, 8, 8)

import matplotlib.pyplot as plt

# print(plt.imshow(digits.images[-1], cmap=plt.cm.gray_r))
# 将scikit应用于这个数据集上，可以将8*8的图像转换为一个长度为64的特征向量
# print("----")
# print(digits.images.shape[0]) # 1797
# data = digits.images.reshape((digits.images.shape[0], -1)) # 列数不知道多少，只是想变为1797行
# print()
# print(data)
