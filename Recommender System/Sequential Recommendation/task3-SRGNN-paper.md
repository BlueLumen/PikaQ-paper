# Session-based Recommendation with Graph Neural Networks

《Session-based Recommendation with Graph Neural Networks》

发现了除了第二篇外，其他几篇都是中国人写的。

前两次收获不太多，毕竟我缺乏方向和方法的基础，所以这次换个方法。

学了一下这个GNN，发现，好像和我研究的那个领域出奇的像，溜了溜了，看看GNN和它有没有什么结合。

# GNN的流程

GNN通过邻接矩阵实现图优化的矩阵化表示。通过聚合操作实现层内和层间的消息传递。
主要参考资料：<br/>
https://www.bilibili.com/video/BV1Tf4y1i7Go<br/>
https://b23.tv/HrBKjvs<br/>



## 一、聚合（Aggregate）

数据就是图，每个节点上都是一个特征。用向量的形式。比如下面这个图。对于A来说，一开始我们并不太确定A的特征到底是啥，所以想通过与A有关系的另外几个节点（邻居）来进行聚合信息。N = aFB+bFC + cFD


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/318c1f09ca734d54a89ce817aa7f9a28~tplv-k3u1fbpfcp-watermark.image?)

层内聚合，似乎也被称为一种池化。邻层之间的聚合的不同，也引出了不同的算法，

## 二、更新

此时A的信息 = δ（W（1,1,1,1,1）+ α * N）

W是权重，α类似于学习率这种。N就是聚合后的信息。δ是激活函数。


## 三、循环

第一次循环


| 节点 | 信息来源 |
| --- | --- |
| A | B、C、D |
|B|A、C|
|C|A、B、C、E|
|D|A、C|
|E|C|

当第二次循环时，A就可以获得E的信息。此时A聚合C时，C就有了上一层聚合的E的特征。

## GNN的目的

通过聚合更新，得到每个点特征。作用的话，就是比如我们要判断A和E有没有关系，那么通过上述方式算出来A和E的特征，拼接起来做分类，用标签做loss，进行优化。
输入：A的特征和Graph的结构，最后获得整个表达。（应该是初始化后的之类的。）

