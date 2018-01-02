# 使用Estimators（评估器）

标签（空格分隔）： tensorflow  机器学习

---

使用tensorflow的第一个例子，使用了sklearn中的分类实验数据集iris（鸢尾花卉数据集），该数据集包含四个特征：花萼和花瓣的高度和尺寸。类别有三类，分别用0,1,2进行处理。

##  读取数据集
```
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 读取数据集，并划分为训练/测试集
iris = datasets.load_iris()

train_feature, test_feature, train_label, test_label = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 0)
```

## 特征选择、搭建模型
建立TensorFlow格式的特征列，iris数据集有4个特征。
```
# 所有特征都是实数值
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]

# 分类器classifier会一直为我们保存训练记录，通过传入特征将模型实例化，设置存储模型训练过程和输出文件的目录model_dir，这样，如果模型训练过程中有中断，可以接着训练
classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    model_dir="/tmp/iris_model")
```

##训练/评估
在准备好数据和模型之后，接下来只需让模型读取数据，再执行训练和测试。为了匹配TensorFlow的数据类型，首先需要定义输入函数。

```
# 输入函数，导入数据
def load_data(split='train'):
            features = {feature_name: tf.constant(train_X)}
            label = tf.constant(train_y)
    return features, label



# 训练（拟合）模型
classifier.train(input_fn=load_data(),steps=1000)

# 评估准确率
accuracy_score = classifier.evaluate(input_fn=input_fn('test'),steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))
```
