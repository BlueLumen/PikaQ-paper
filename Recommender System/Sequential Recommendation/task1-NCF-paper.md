# Neural Collaborative Filtering

不采用那种每段的了。效率比较低，直接看一遍，脑袋里面剩下啥就是啥。

实验部分没看，因为队长也没看。现在的主要目的是，明白总体原理和整体逻辑即可，实验部分对我可能意义不大，就算有那也无所谓了，毕竟我就是没看。

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

## 3.2 Generalized Matrix Factorization (GMF)

在这一小节，作者设计了一个公式，能够将NCF融入进MF框架里，这也称之为广义MF。具体见下面这个图，通过对h和a进行特定的赋值（分别是单位向量和等值函数），就可以将这玩意退化为MF。相当于是凑了个推广形式？Maybe~


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ca9d7f804bda4dda95cc1f89d019775c~tplv-k3u1fbpfcp-watermark.image?)

## 3.3 MLP

顾名思义，这部分就讲了多层感知机以及其参数。先说MF的向量内积形式忽视了很多交互信息啥的，所以多层感知机非常好。

然后用公式刻划了一下，也比较好理解。讲了W、b、a分别的含义，然后分析了三种激活函数的优缺点和适用性，最后选择了ReLU。然后根据经验设计双塔模型，和层数等等。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c5976fd6388640598edcc9b2e20c4402~tplv-k3u1fbpfcp-watermark.image?)

## 3.4 Fusion

融合一下。这一部分就讲了MLP和GMF如何统一起来。是合并还是分开。讨论了优缺点，最后采用分开的形式，用最后的隐藏层进行串联。

下面这个图就是形式化的公式，然后参数们都可以通过算偏导反向传播进行计算。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4bb48517de3542ae9f0f4fd2d8fc1532~tplv-k3u1fbpfcp-watermark.image?)

