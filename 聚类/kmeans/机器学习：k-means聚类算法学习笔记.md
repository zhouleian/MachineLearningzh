﻿# 机器学习：k-means聚类算法学习笔记

标签（空格分隔）： 机器学习

---

# 一、应用背景
前面介绍了k-近邻，决策树，逻辑斯蒂回归，AdaBoost算法等监督学习算法，这里介绍无监督学习算法：k-means算法。

监督学习算法和无监督学习算法的区别在于：
1. 监督算法要求必须事先明确知道各个类别的信息，数据样本必须有一个类别与之对应。常见应用为分类。
2. 无监督学习，学习过程中训练样本没有类别信息，只处理特征，不操作监督信号。在无监督学习中研究最多，训练最广的是聚类算法。

然而，在实际任务中，很多情况下无法得到已经标记类别的训练样本，尤其是在处理海量数据的时候，如果通过预处理使得数据满足分类算法的要求，则代价非常大，这时候可以考虑使用聚类算法。

聚类算法作为无监督学习，目标是发现样本特征 $\vec{x}$ 隐含的类别标签 y。将相似的对象归到同一个簇中（可近似看做类别），将不相似对象归到不同簇中，簇内对象越相似，聚类效果越好。而相似的判断则取决于选择的相似度计算方法。聚类任务的完成只是将数据集分成了很多簇，簇所对应的概念和语义需要使用者来把握和命名。

本文介绍聚类算法的一种——k-means算法。后文重点讨论k-means算法的数学基础，算法步骤以及应用场景，并介绍一种更有效的二分k-means算法。

# 二、k-means算法基本概念
k-means算法是一种聚类算法，目标是发现训练数据集的k个簇，这里的k是用户给定的。
这里介绍“簇”是什么，如何描述“簇”。

# 2.1 簇
k-means的簇可类似看做“类别”，k-means算法通过选择的相似度计算方式，将相似的对象归到同一个簇中，将不相似对象归到不同簇中。每一个簇通过其“质心”描述。
# 2.2 质心
质心，描述簇，是簇中所有点的中心。


# 三、 k-means算法
k-means算法是一种聚类算法，目标是发现训练数据集的k个簇，这里的k是用户给定的。

下面以一个例子介绍k-means算法。
![此处输入图片的描述][1]。

## 3.1 算法介绍

训练集中每个数据样本用一个n维特征向量描述n个属性的值，即 X = {x1，x2，...，xn}，用户给定聚类k簇，分别用c1,c2,...,ck表示。训练数据集共N个样本，则
>输入：训练集D={(x<sub>1</sub>,y<sub>1</sub>),(x<sub>2</sub>,y<sub>2</sub>),…,(x<sub>m</sub>,y<sub>m</sub>))} 
    聚类簇数 k
    
> 输出：簇划分C={C1,C2,...Ck}

$ $
>过程：
1) 从数据集 $D$ 中随机选择 $k$ 个样本作为初始的 $k$ 个质心向量： $\lbrace {μ_1,μ_2,...,μ_k}\rbrace$
$ $
2）对于n=1,2,...,N
$ $
a) 将簇划分C初始化为 $C_t=∅，t=1,2...k$
b) 对于 $i=1,2...m$ ,计算样本 $x_i$ 和各个质心向量 $μ_j$ (j=1,2,...k)的距离：
将 $x_i$ 标记最小的距离为 $d_{ij}$ 所对应的类别 $λ_i$。此时更新 $C_{λ_i} = C_{λ_i} ∪   { x_i}$
$ $
c) 对于 $j=1,2,...,k,$对 $C_j$ 中所有的样本点重新计算新的质心
$ $
e) 如果所有的k个质心向量都没有发生变化，（簇不发生变化，或者达到最大迭代次数，或者目标函数达到最优）则转到步骤3）
$ $
3） 输出簇划分

### 3.1.1 算法结束条件
1. 某次迭代后簇不发生变化（或者只有很少几个样本点更新所属簇，这时可认为已经达到了收敛），或者
2. 达到最大迭代次数，或者
3. 目标函数达到最优
在计算样本点到质心的距离时，常采用欧几里得距离或余弦相似度。当采用欧式距离时，目标函数一般为最小化对象到其簇质心的距离的平方和，如下：
$$ min\sum^K_{i=1}\sum_{x\inC} dist(c_i,x)^2$$
当采用余弦相似度时，目标函数一般是最大化对象到其簇质心的余弦相似度和，如下：
$$ min\sum^K_{i=1}\sum_{x\inC} cosine(c_i,x)$$

### 3.1.2 k值选择
k-means算法首先要指定 k 值，一般会根据对数据的先验知识选择一个合适的k值，如果没有什么先验知识，则可以通过交叉验证选择一个合适的k值。
### 3.1.3 初始化质心
上面的算法是随机选择k个质心，但是质心的选择对最后的聚类结果和运行时间都有很大的影响，因此需要选择合适的k个质心，最好这些质心不能太近。
针对质心的选择，有很多改进的算法，而不是随机选择：
1. K-Means++

a)  从输入的数据点集合中随机选择一个点作为第一个聚类中心 μ1

b) 对于数据集中的每一个点xi，计算它与已选择的聚类中心中最近聚类中心的距离
$D(x)=argmin∑_{r=1}^{k_{selected}}||x_i−μ_r||_2^2$

c) 选择一个新的数据点作为新的聚类中心，选择的原则是：
D(x)较大的点，被选取作为聚类中心的概率较大
d) 重复b和c直到选择出k个聚类质心
e) 利用这k个质心来作为初始化质心去运行标准的K-Means算法
2. 先进行层次聚类得到k个簇，将这k个簇的质心作为初始质心，但层次聚类开销较大，所以只适合于：（1）样本相对较小，例如数百到数千）；（2）K相对于样本大小较小

### 3.1.4 迭代
上述算法中，需要计算每一个样本点和每一个质心的距离，收敛缓慢，有相应的改进算法：距离计算优化  elkan K-Means 算法，大样本优化Mini Batch K-Means算法等。

### 3.1.5 聚类效果评价
在监督学习中，测试样本有标签，我们可以通过错误虑或准确率，召回率来评估算法。而无监督学习中，没有比较直接的聚类评估方法。

直观上看，我们希望聚类之后“物以类聚”，同一簇中样本尽可能相似，尽可能稠密，不同簇中样本尽可能不同和离散，即“簇间相似度低，簇内相似度高”。常见的方法有轮廓系数 $Silhouette Coefficient$ 和 $Calinski-Harabasz Index$ 。得到的 $Silhouette Coefficient$ 和 $Calinski-Harabasz Index$ 分数值越大则聚类效果越好。
　　　　
1. Calinski-Harabasz分数值s的数学计算公式是：
$$s(k) = \frac{tr(B_k)}{tr(W_k)}\frac{m−k}{k−1}$$
其中m为训练集样本数，k为簇数。$B_k$为簇之间的协方差矩阵，$W_k$为簇内部数据的协方差矩阵。tr为矩阵的迹。
也就是说，类别内部数据的协方差越小越好，类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。

在scikit-learn中， Calinski-Harabasz Index对应的方法是metrics.calinski_harabaz_score。

2. 轮廓系数计算公式：
第 i 个对象的轮廓系数
$$ S(i) = \frac{b(i)-a(i)}{max\lbrace{a(i),b(i)}\rbrace} $$
$a_i$ 为第 i 个对象，计算它到所属簇中所有其他对象的平均距离，记 ai （体现凝聚度）
$b_i$ 为 对于第 i 个对象和不包含该对象的任意簇，计算该对象到给定簇中所有对象的平均距离，记 bi （体现分离度）

轮廓系数取值为[-1, 1]，其值越大越好，且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。对于接近0的结果，则表明聚类结果有重叠的情况。

在scikit-learn 中，对应的方法是 sklearn.metrics.silhouette_score。


## 3.2 算法步骤 

k-means 算法的基本流程是：

![此处输入图片的描述][2]

1. `收集数据`：采用任意方法收集
2. `准备数据`：由于需要进行距离计算，因此要求数据类型为`数值型`。

3. `分析数据`：采用任意方法对数据进行分析
4. `训练算法`：k-means算法作为无监督聚类算法，**没有训练过程**。

5. `测试算法`：应用聚类算法，观察结果。使用量化误差指标，评价算法聚类结果。
6. `使用算法`：可用于所希望的任意应用，通常，簇质心可以代表整个簇的数据来做出决策。

## 3.3 算法优缺点

`优点`：容易实现 ，收敛速度快。需要调整的参数仅仅是k。

`缺点`：k值得选取，质心的选择。迭代方式可能收敛到局部最小值，在大规模数据集和不是凸数据集上收敛较慢；对噪音和异常点比较的敏感。

如果各隐含类别的数据不平衡，聚类效果差。比如，各隐含类别的数据量严重失衡，或者各隐含类别的方差不同，则聚类效果不佳。

`适用数据类型`：数值型数据

# 四、实例

对于上面给出的西瓜数据集，执行 $k-means$ 算法，假设簇数 k = 3，算法开始随机选择 k 个数据点作为质心，这里选择了 $x_6,x_{12},x_{27}$ 作为初始质心,则3个初始均值向量为 

$μ_1 = （0.403，0.237）$
$μ_2 = （0.343，0.099）$
$μ_3 = （0.532，0.472）$

下面则是算法步骤（2），计算数据集中每一个数据点和这3个质心的距离，这里以 $x_1 = (0.697,0.460)$ 为例:

计算 $x_1$ 与均值向量 $μ_1,μ_2, μ_3 $ 的距离分别为：0.369,0.506，0.166。所以，$x_1$和均值向量$μ_3$距离最近，准备划分到簇 $c_3$中。

对数据集中每一个样本进行类似处理之后，得到当前簇划分：
$C_1 = \lbrace{x_5,x_6,x_7,x_8,x_9,x_{10},x_{13},x_{14},x_{15},x_{17},x_{18},x_{19},x_{20},x_{23}}\rbrace$
$C_2 = \lbrace{x_{11},x_{12},x_{16}}\rbrace$
$C_3 = \lbrace{x_1,x_2,x_3,x_4,x_{21},x_{22},x_{24},x_{25},x_{26},x_{27},x_{28},x_{29},x_{30}}\rbrace$

于是，由这3个新的簇得到新的 均值向量：

>均值向量 = $\frac{C_i的特征j和}{C_i样本个数}$，
对$μ_2$包含的3个数据（分母为3）做示例

$μ_1 = （0.473，0.214）$
$μ_2 = (\frac{0.245+0.343 + 0.593}{3} ,\frac{0.057 + 0.099 + 0.042}{3})=（0.394，0.066）$  
$μ_3 = （0.623，0.388）$

不断重复这样的过程：计算每个样本的簇，计算新的均值向量，第五轮迭代结果和第四轮迭代结果相同，所以，算法停止，得到最终簇划分。

# 五、sklearn k-means实例
## 5.1 k值的选择
### 5.1.1 畸变函数
$$ J(c,u) = \sum_{i=1}^m||x^{(i)} - u_{c^{(i)}}||^2$$
表示每个样本点到其质心的距离平方和，畸变函数也叫做成本函数。每个簇的畸变程度等于该簇质心和簇内样本点的距离的平方和，所以，成本函数是各个簇的畸变程度（distortions）之和。
一个簇内部样本点越稠密，簇的畸变程度越小，反之，越离散，畸变程度越大。理想的聚类就是求解成本函数最小化的参数，即一个重复配置每个类包含的观测值，并不断移动类重心的过程。

### 5.1.2 肘部原则
肘部原则是畸变函数的应用，k值得选择对聚类结果影响很大，可以通过肘部法则来估计聚类数量。随着k值的增大，平均畸变程度会减小；每个类包含的样本数会减少，于是样本离其重心会更近。但是，随着 值继续增大，平均畸变程度的改善效果会不断减低。值增大过程中，畸变程度的改善效果下降幅度最大的位置对应的 k 值就是肘部。

```
def get_k():
    dataSet,x1, x2 = loadDataSet('testSet.txt')
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(
            X, kmeans.cluster_centers_, "euclidean"), axis=1)) / X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度')
    plt.title('best K')
    plt.show()
```
![肘部原则][3]
所以，根据肘部原则得到最好的k值为：4。

## k-means应用
```
def kmeans_test():
    # 原始数据
    dataSet,x1,x2 = loadDataSet('testSet.txt')
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    plt.figure(figsize=(15, 20))
    p1 = plt.subplot(3, 2, 1)
    p1.scatter(x1, x2, c='b',marker='o')
    # 测试多个k
    tests = [2, 3, 4, 5, 8]
    subplot_counter = 1
    for k in tests:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter) # 占据第subplot_counter个位置
        kmeans_model = KMeans(n_clusters=k, random_state=9).fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans_model)
        # 使用Calinski-Harabasz分数值s比较
        plt.title('K = %s, C = %.03f' %
                  (k,metrics.calinski_harabaz_score(X, kmeans_model)))
      
    plt.show()

```
![此处输入图片的描述][4]

可以看到 当 k= 4 时，聚类效果最好，这时的  Calinski-Harabasz分数值s也是最大的。

  [1]: http://omxy7x542.bkt.clouddn.com/17-12-20/90156199.jpg
  [2]: http://omxy7x542.bkt.clouddn.com/17-12-20/69162697.jpg
  [3]: http://omxy7x542.bkt.clouddn.com/17-12-23/16980285.jpg
  [4]: http://omxy7x542.bkt.clouddn.com/17-12-23/47831242.jpg