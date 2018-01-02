import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics


from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，
# 共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]，
# 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2,
                  centers=[[-1,-1], [0,0], [1,1], [2,2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state =9)
# print(X)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

# iris = datasets.load_iris()  # 加载iris数据集
# digits = datasets.load_digits()  # 加载digits数据集 ,手写识别
# print(iris)
# print(digits)
# print('digits.data:', digits.data) # 用来分类样本的特征
# print('digits.target:', digits.target)  # 给出了digits数据集的真实值，
# 就是每个数字图案对应的想预测的真实数字

# print('iris.data:', iris.data)
# print('iris.target:', iris.target)

# print(digits.images[0])
# print(digits.images)
# 每个数据集都与标签对应，使用zip()函数构成字典
# images_and_labels = list(zip(digits.images, digits.target))

def makeplt():
    for index, (image, label) in enumerate(images_and_labels[:4]):
        print(index)
        print(image)
        print(label)
        plt.subplot(2, 4, index + 1)
        plt.axis('off') # 图像坐标轴是否出现，off不出现
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training:%i' % label)
        # plt.show()

# 手写识别
def test():
    classifier = svm.SVC(gamma=0.001)  # svm预测器

    classifier.fit(data[:1000], digits.target[:1000])
    # classifier.fit(data[:(n_samples /2)], digits.target[:(n_samples / 2)])
    # 使用数据集的一半进行训练数据

    expected = digits.target[1000:]
    predicted = classifier.predict(data[1000:])  # 预测剩余的数据
    # expected = digits.target[n_samples / 2:]
    # predicted = classifier.predict(data[n_samples / 2:])  # 预测剩余的数据

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    images_and_predictions = list(zip(digits.images[:1000], predicted))
    # 图片与预测结果按照字典方式对应
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):# 预测四个
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')  # 展示图片
        plt.title('Prediction: %i' % prediction)  # 标题
        plt.show()

def svmtest():
    clf = svm.SVC(gamma=0.001, C=100.)
    print(clf.fit(digits.data[:-1], digits.target[:-1]))  # 对前面所有的数据进行训练
    print(clf.predict(digits.data[-1:])) # 对最后一个数据进行预测


# if __name__=='__main__':
#     digits = datasets.load_digits()  # 加载digits数据集 ,手写识别
#     images_and_labels = list(zip(digits.images, digits.target))
#     n_samples = len(digits.images)  # 样本的数量
#     #  print(n_samples)
#     data = digits.images.reshape((n_samples, -1))
#     svmtest()

    # print(len(data))
    # test()
    # aa = np.array([1, 2, 3, 4])
    # ab = aa.reshape((4,-1))
    # print(type(aa))
    # print(type(ab))
    # if aa==ab:
    #     print(1)
    # else:
    #     print(0)
