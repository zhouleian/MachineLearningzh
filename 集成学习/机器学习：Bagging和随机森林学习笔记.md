﻿# 机器学习：Bagging和随机森林学习笔记

标签（空格分隔）： 机器学习

---

# 一、应用背景

前面介绍了集成学习，集成学习方法按照个体学习器之间是否存在依赖关系可以分为两类：
1. 第一个是个体学习器之间存在强依赖关系，必须串行执生成的序列化方法：代表算法是Boosting算法； 上篇介绍。
2. 第二个是个体学习器之间不存在强依赖关系，可同时生成的并行化方法，代表算法是bagging和随机森林（Random Forest）算法。

本文主要介绍bagging和随机森林算法，重点讨论bagging和随机森林算法基本概念和实现步骤。


# 二、Bagging算法基本概念
Bagging算法是一种基于数据随机重抽样的分类器构建方法，是并行式集成学习的著名代表。

AdaBoost和Bagging都是集成学习方法，同样也要解决集成学习中的两个关键问题：
 1. 如何产生每个分类器？在每一轮如何改变训练数据的权值或概率分布；
 2. 如何结合各个弱学习器组合得到强分类器？

下面介绍Bagging算法，解释Bagging是如何解决这两个关键问题的。
 
## 2.1 算法介绍
Bagging算法的提出是基于一个假设“基学习器的误差相互独立”，在这个假设之下，我们可以得到最好的训练结果，集成的错误率将趋于0，但是这个假设一般很难满足，因为所有的基学习器是为解决同一个问题训练出来的，显然是不可能独立的。所以，退而求其次，既然“独立”无法做到，我们可以设法使得基学习器尽可能有较大的差异。

一种可能的做法是对训练数据进行随机采样得到若干个采样集，再从每个采样数据集中训练得到弱分类器，这样，由于训练数据不同，我们获得的采样集不同，就使得基学习器有了较大的差异。

然而，我们还希望基学习器不会太差，如果采样得到的每个数据集都不同，则每个基学习器只使用了一小部分的训练数据，甚至不足以进行有效学习，这显然不能保证我们能够得到好的基学习器。

所以，**考虑使用相互有交集的采样子集**。

![Bagging][1]


1. 首先，从原始训练数据集，通过随机采样T次得到T个采样集，采样集和原始数据集大小一样；

2. 对这T个采样集训练得到T个弱学习器；

3. 对要分类的新数据，应用T个分类器进行分类，投票表决，得到最终的分类结果；回归任务时，使用简单平均法得到结果。

**`随机采样`**：
一般采用的是自助采样法（Bootstap sampling）,即对于m个样本的原始训练集，每次先随机采集一个样本放入采样集，接着把该样本放回，也就是说下次采样时该样本仍有可能被采集到，即允许采样集中有重复的值。

这样采集m次，最终可以得到m个样本的采样集，由于是随机采样，这样每次的采样集是和原始训练集不同的，和其他采样集也是不同的，这样得到多个不同的弱学习器。
　　　　
**`投票表决`** ：
解决了集成学习的第二个关键问题，具体的说：选择分类器投票结果中类别最多的类作为最后的分类结果。

## 2.1 算法流程
Bagging算法的过程要简单很多，给定二分类训练集，

> 输入：
训练数据集 D={(x<sub>1</sub>,y<sub>1</sub>),(x<sub>2</sub>,y<sub>2</sub>),…,(x<sub>N</sub>,y<sub>N</sub>))} 
其中，x<sub>i</sub>∈X⊆R<sub>N</sub>为实例的特征向量，yi∈Y={1，-1}为实例的类别，i=1,2,…,N ;；

$ $
> 输出：
最终分类器 G(x)

**过程：**

1）对于$t=1,2...,T$:
　a)对训练集进行第t次（共T次，T个弱分类器）随机采样，共采集m次，得到包含m个样本的采样集$ D_m$
　b)用采样集 $ D_m $ 训练第m个弱学习器 $G_m(x)$

2) 如果是分类算法预测，则投票表决得到最终类别。如果是回归算法，T个弱学习器得到的回归结果进行算术平均得到的值为最终的模型输出。

## 3.2 算法步骤 

Bagging算法的基本流程是：


1. `收集数据`：采用任意方法收集
2. `准备数据`：依赖于所使用的分类器类型

3. `分析数据`：采用任意方法对数据进行分析
4. `训练算法`：Bagging大部分时间将用于训练

5. `测试算法`：计算分类错误率。
6. `使用算法`：预测类别。



## 3.3 算法优缺点
`优点`：每次都进行采样来训练模型，因此泛化能力很强，对于降低模型的方差很有作用。

`缺点`：对于训练集的拟合程度会差一些，也就是模型的偏倚会大一些。

`适用数据类型`：数值型和标称型数据（即分类离散值）

# 四、随机森林RF

随机森林是 Bagging 的一个扩展变体， RF 和 Bagging 只有两点不同，其他相同，下面介绍RF的基本概念和算法步骤，以及相比 Bagging 的优点。

## 4.1 随机森林的基本概念
RF和Bagging的基本思想相同，并没有脱离Bagging的范畴，只是有两点不同：

1. 随机森林的弱学习器都是决策树；

2. RF在Bagging的样本随机采样基础上，进一步在决策树的训练过程中加上了特征的随机选择。

第一点很容易理解，这里解释第二点：

传统决策树在/选择划分属性时,是在当前节点的属性集合（假设有d个属性）中选择一个最优属性; 而在RF中, 对基决策树的每个节点, 先从当前节点的属性集合中随机选择一个包含k个属性的子集， 再从k个中选择一个最优属性用于划分。

这里的参数k控制了随机性的引入程度. 一般情况下推荐值 $ k = log_2d $。当 k = d 的时候，与传统决策树相同；当k = 1的时候，即随机选择一个属性进行划分。

## 4.2  RF算法流程

给定二分类训练集，

> 输入：
训练数据集 D={(x<sub>1</sub>,y<sub>1</sub>),(x<sub>2</sub>,y<sub>2</sub>),…,(x<sub>N</sub>,y<sub>N</sub>))} 
其中，x<sub>i</sub>∈X⊆R<sub>N</sub>为实例的特征向量，yi∈Y={1，-1}为实例的类别，i=1,2,…,N ;

$ $
> 输出：
最终分类器 G(x)

**过程：**

1）对于$t=1,2...,T$:
　a)对训练集进行第t次（共T次，T个弱分类器）随机采样，共采集m次，得到包含m个样本的采样集$ D_m$
　b)用采样集 $ D_m $ 训练第m个弱学习器 $G_m(x)$，这里和bagging不同的是：在所有的样本属性集中，选择包含k个属性的子集，从k个属性中选择一个最优属性进行划分。

2) 如果是分类算法预测，则投票表决得到最终类别。如果是回归算法，T个弱学习器得到的回归结果进行算术平均得到的值为最终的模型输出。


## 4.3 RF优点

随机森林简单, 易实现,计算开销小,令人惊奇的是,它在很多现实任务中展现出强大的性能,被誉为”代表集成学习技术水平的方法”。

Bagging采用随机采样的目的是，提高基分类器的多样性，降低集成错误率。RF对Bagging做了小小的改动，不止采用“样本扰动”来增加基分类器的多样性，还加入了“实行扰动”的特性，使得最终集成的泛化性能可通过个体学习器之间的差异度进一步增加。

这种特性使得，随着基学习器的增加，随机森林通常比Bagging算法收敛到更低的泛化误差。在决策树基分类器构建的时候，Bagging使用了“确定型”的决策树，对所有属性进行了考察，RF“随机型”决策树，只考察了一个属性子集。




  [1]: http://omxy7x542.bkt.clouddn.com/17-12-18/977026.jpg
 