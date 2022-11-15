# Neural Collaborative Filtering

不采用那种每段的了。效率比较低，直接看一遍，脑袋里面剩下啥就是啥。

参考引用：
1. latent features：https://www.zhihu.com/question/306016801?sort=created<br/>
2. latent features：[machine learning - Meaning of latent features? - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/749/meaning-of-latent-features)
3. pointwise、pairwise等：https://zhuanlan.zhihu.com/p/322065156
4. Loss log：https://www.cnblogs.com/klchang/p/9217551.html

# Abstract

简单讲了讲深度学习在除推荐外的领域都很风靡了。但是在推荐领域热度还不大。即使有应用，但也不温不火如隔靴搔痒。所以作者尝试将深度学习应用在协同过滤上。

# 1. Introduction

1. 传统协同过滤方法涉及到一个MF，还有一些内积的操作。深度学习在推荐领域也有点应用，但都是对辅助信息进行建模的，比如商品文字描述巴拉巴拉一大堆。

2. 作者尝试对 非评分和评论的其他隐形特征进行建模。当然这样导致用户对商品满意情况更难观察，因此会缺少很多负类标签。于是作者在这个论文中探索了一种方式，去用DNN进行建模。

> We focus on implicit feedback, which indirectly reflects users’ preference through behaviours like watching videos, purchasing products and clicking items

# 2. PRELIMINARIES

## 2.1 Learning from Implicit Data

作者讲了一下协同过滤的矩阵形式，并提出了自己的建模思路。几个关键词是：

> The recommendation problem with implicit feedback is formulated as the problem of estimating the scores of unobserved entries in Y, which are used for ranking the items.

> pointwise loss and pairwise loss

> Formally, they can be abstracted as learning ˆyui = f(u, i|Θ), where ˆyui denotes.the predicted score of interaction yui, Θ denotes model parameters, and f denotes the function that maps model parameters to the predicted score (which we term as an interaction function)

## 2.2 Matrix Factorization

这一部分先介绍了什么是 MF，然后举了几个例子（见下图）来证明了MF里的内积方法在低维空间（特征较少）不是那么的准确，而高维空间下呢，又存在着过拟合的危险。所以作者引入深度神经网络来对交互函数进行建模。

> possible limitation of MF caused by the use of a simple and fixed inner product to estimate complex user–item interactions in the low-dimensional latent space.

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/13897f8762a945849bc6eb4319382bea~tplv-k3u1fbpfcp-watermark.image?)

# 3 NCF

先提出一般的NCF模型。然后再实例化，用多层感知机去学习user-item交互函数。最后在NCF的框架下，提出了一个NMF模型，借助MLP非线性性质和MF的线性性质来 modelling the user–item latent structures.

## 3.1 General Framework

讲了基本的框架。提到了，embedding 可以看作 latent vector。然后介绍了这个网络的结构。三天前看不懂，现在看懂了。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/35060dccbc294e5c82cd616aff73aac1~tplv-k3u1fbpfcp-watermark.image?)

### 3.1.1 Learning NCF

先说最后一段，作者对采样方法画了个饼。

这一部分，实际就是讲了一下训练过程，先抛出来损失函数公式，训练集用的是啥（正例和负例），以及其分别是啥。提到了极大似然法，随机梯度下降等等。私以为，有点类似于BP神经网络算法。作者说这个方法就是 loss log，算是loss log。let me 搜搜啥是 loss log。

好了，搜到了，就是对数损失函数，南瓜书学过了，和交叉熵的那个等价的东西。终于知道了，论文中写到的 probabilistic treatment是啥玩意了，原来就是这东西。
