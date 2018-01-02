# reshape，plot

标签（空格分隔）： 未分类

---

## reshape
在对数据集中处理或者选择上，通常会遇到reshape函数，
reshape(n,m),通常n，m的值为正整数，也可以为-1。下面用例子来解释：（reshape_test.py）
```
z = np.array([[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]])
print(z.shape)
// 很明显，(4, 4)
```
以z为例，介绍reshape(n,m)函数用于改变数组的形状
```
zz = z.reshpe(2,8)
print(zz)
//[[ 1  2  3  4  5  6  7  8]
// [ 9 10 11 12 13 14 15 16]]
//这时，z数组不变。即reshape不改变原来数组的大小
```
上面说的是n,m均为正整数的情形，新的数组zz的大小必须和原来z的大小一样，否则出错，有时n或m会取值为-1：

1. 有时候不知道z的属性是多少，想让z变成只有i行或i列的数组，这时候，就需要 -1,Numpy会自动计算有多少列或多少行。
2. reshpae(-1,i) 数组转化为有i列，自动计算有多少行
3. reshpae(i,-1) 数组转化为有i行，自动计算有多少列
```
// 变成只有一行
zz = z.reshape(-1)
print(zz)

//array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])

//只有一列
zz = z.reshape(-1,1)
print(zz)

//array([[ 1],
        [ 2],
        [ 3],
        [ 4],
        [ 5],
        [ 6],
        [ 7],
        [ 8],
        [ 9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16]])

// 两列，行数自动计算
z.reshape(-1, 2)
//array([[ 1,  2],
        [ 3,  4],
        [ 5,  6],
        [ 7,  8],
        [ 9, 10],
        [11, 12],
        [13, 14],
        [15, 16]])
```

## plot画图
### 数据曲线
```
//两条曲线在同一个图中，也可以是两个图
#定义两个函数，画图
def f1(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

def f2(t):
    return np.sin(2 * np.pi * t) * np.cos(3 * np.pi * t)

def tt():
    t = np.arange(0.0, 5.0, 0.02)
    plt.figure()
    plt.plot(t, f1(t), "g-", label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
    plt.plot(t, f2(t), "r-.", label="$g(t)=\sin (2 \pi t) \cos (3 \pi t)$", linewidth=2)

    plt.axis([0.0, 5.01, -1.0, 1.5])
    plt.xlabel("t")
    plt.ylabel("v")
    plt.title("example")

    plt.grid(True)
    plt.legend()
    plt.show()
```
![曲线图][1]

在画图的时候，也可能会两个曲线在同一个图中，这时候用p1,p2分别表示两个曲线。用到subplot
```
p1 = plt.subplot(211) # 211，将图分为,2行1列，p1是第一个
p2 = plt.subplot(212) # 212 ，2行1列，p2是第二个
```

### plot 画散点图
```
//plot.py

//导入pyplot，通常设置别名：plt
import matplotlib.pyplot as plt
//创建一幅图，是有一个Figure()对象的返回值的
plt.figure()

//根据x,y的值将数据画成曲线，显示，x就对应横坐标，y对应纵坐标，x，y都是一个一维的list。还可以指定线的样式，可以是虚线，点线，还有颜色，线的宽度，这些可以使用关键字参数指定。
plt.plot(x,y)

//显示
plt.show()

```
```
//针对西瓜数据集的密度和含糖率的图，分别读取好瓜（蓝色）和不是好瓜（红色）数据画图，并且设置图标题和坐标名称。

dataSet, classLabels,x,y = loadDataSet('xiguahao.txt')

fig = plt.figure()
ax1 = fig.add_subplot(111)
# 设置标题
ax1.set_title('Lemon plot')
# 设置X轴标签
plt.xlabel('Density')
# 设置Y轴标签
plt.ylabel('Sugar content')
# 画散点图
ax1.scatter(x, y, c='b', marker='o')
dataSet2, classLabels2,x2,y2 = loadDataSet('xiguahuai.txt')
ax1.scatter(x2, y2, c='r', marker='o')
plt.legend('ab')
# 设置图标
# 显示所画的图
 plt.show()
```
![此处输入图片的描述][2]

### 图中带文字
有时候需要在图中添加文字，即标注，通过pyplot.text,由pyplot或者subplot调用。
```
text(tx,ty,fontsize=fs,verticalalignment=va,horizontalalignment=ha,...)
```
其中，tx和ty指定放置文字的位置，va和ha指定对其方式，可以是top，bottom，center或者left，right，center,还可以使文字带有边框，边框形状还可以是箭头，并指定方向。

### 箭头
有时需要在图中使用箭头指明某点或某条曲线，需要用到箭头，pyplot中的箭头使用使用pyplot.annotate，调用方式与text类似。
```
annotate(text,xy=(tx0,ty0),xytext=(tx1,ty1),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
```
其中，text是与箭头一起的文字，xy是箭头所在位置，终点，xytext是起点（箭头，起点<——终点），arrowtypes指定箭头的样式等。

```
# 文本标注
p.text(tx, ty, label2, fontsize=15, verticalalignment="bottom",horizontalalignment="left")

# 箭头，起点<——终点
p.annotate('', xy=(1.8, 0.5), xytext=(tx, ty),arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
```
## 局部放大
顾名思义，局部放大观察。
![局部放大][3]
局部放大需要4个步骤：

1. subplot分为两部分，左图是总图，右图是放大之后的图，画出两部分的图。

2. 在左图中画方框，指定四个顶点，依次画出四条线段，从某个点绕个圈再回到起点，就画出了方框，使用pyplot(x,y)，可以方便地指定颜色，线宽等。
```
# 在左边图中框出要放大的局部区域
    tx0 = 4
    tx1 = 4.5
    ty0 = -0.1
    ty1 = 0.1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    p1.plot(sx, sy, "blue")
```

3.画两条跨子图的线（连接线），使用matplotlib.patches有一个ConnectionPatch类型，它可以用在一个或多个子图之间画线
```
con = ConnectionPatch(xyA=xy1,xyB=xy0,coordsA="data",coordsB="data",
axesA=p1,axesB=p0)
p1.add_artist(con)
```
这里xyA是p1里面的点，xy0是p0里面的点，coordsA和coordsB默认值"data"，也不用改，然后就是axesA,要添加ConnectionPatch的子图（左图），axesB，要连接的子图（右图）。

4.最后使用p1.add_artist(con)，将连接线添加进子图。




  [1]: http://omxy7x542.bkt.clouddn.com/17-12-23/12661889.jpg
  [2]: http://omxy7x542.bkt.clouddn.com/17-12-23/73327052.jpg
  [3]: http://omxy7x542.bkt.clouddn.com/17-12-23/46979893.jpg