# paper 乱读
《Session-based recommendations with recurrent neural networks》

# Abstract 

作者介绍了研究背景：在session-based的推荐系统情景中，被大家所称赞的矩阵分解方法并不准确。在实践中往往通过I2I的方式来解决，例如推荐相似的items。作者断言，可以对整个session过程建模，就可以使得推荐更加准确。因此作者引入了基于RNN的方法。在传统RNN方法的基础上做出了一些改进，并用实验验证了该方法的结果取得SOAT。

# 1.Introduction

1. 第一段介绍了基于对话的推荐系统一段时间来不受重视的原因以及当时的现状。
2. 第二段介绍了两个流派的方法：**factor** 和 **neighbor**，前者不适合在对话推荐系统中应用，后者更适合。
3. 第三段开始引入故事的主人公，**RNN**，先介绍了它在其他领域的风靡，然后提到它处理非结构化数据的方法备受关注。
4. 第四段开始将 **RNN** 引入推荐系统。作者引入了一个新的ranking loss function。同时作者谈到基于对话的推荐系统和NLP在某些方面有相似之处（as long as they both deal with sequences）。之后又巴拉巴拉了一大些，几个关键词就是 input/output，click，ranking loss function以及large item set and large click-stream datasets。

# 2. Related Work
## 2.1 Session-based recommendation

1. 第一段概述了会话系统中的 **I2I** 方法。这种方法虽然简单，却也很有效并被广泛应用。但是有一个问题是该方法只考虑用户的最后一次点击，而 ignores the information of the past clicks 
2. 第二段介绍了马尔可夫决策过程在基于对话的推荐系统的应用及优势和劣势。
3. 第三段介绍了GFF方法。

## 2.2 Deep learning in recommenders
1. 就一段，介绍了最早的一个神经网络相关的方法 RBM for Collaborative Filtering，这个方法（模型）现在也是最好的协同过滤相关的算法（模型）之一。然后讲了一个卷积神经网络提content-feature的方法，以及其适用的情景。

# 3. Recommendations with RNNs

1. 第一段，RNNs被设计用于对可变长度序列数据进行建模，和卷积前馈深度模型的主要区别在于一个隐藏层啥的。然后介绍了一个对于隐藏层的更新函数。
2. 引入 **GRU** ，它可以处理梯度消失的问题。总之介绍了 **RNN** 和 **GRU** 的一些广泛的特性。

## 3.1 Customizing the GRU model

1. 第一段，作者在session-based推荐模型中使用了基于 **GRU** 的 **RNN**。
2. 第二段和后续合并了，见下面这些英文提炼。先采用不求甚解的策略通读一下。

- Input of network：actual state of the session
- output of network：item of next event in the session
- state of session：item of actual event or event in session so far
- output：the predicted preference of the items i.e. the likelihood of being next in the session for each item
- hidden state of previous layer：when multiple GRU layer used, it can be input of next one



![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/354cc44d41754fb68def127b82c407c0~tplv-k3u1fbpfcp-watermark.image?)

### 3.1.1 Session-parallel mini-batches

1. 第一段，讲了 **RNNs** 处理 **NLP** 问题的一个操作：**in-sequence mini-batches**，其中采用了滑窗策略。但是这个方法不适合 **session Recommendation** ，并介绍了原因。于是引入了 **Session-parallel mini-batches**。（sessions之间独立，巴拉巴拉）
2. 第二段，捋一下这个方法，见下述无序列表。私以为， **i1,1** 第一个下表为session，第二个下标为event。哎咱就是说，不对咱也没办法。


- 第一，将session排序
- 第二，use the first event of the first X sessions to form the input of the first mini-batch
- the desired output is the second events of our active sessions
- 然后，the second mini-batch is formed from the second events and so on


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8c3a145ed27d4373b390099f89c8b250~tplv-k3u1fbpfcp-watermark.image?)





