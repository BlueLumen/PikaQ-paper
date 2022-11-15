# ImageNet Classification with Deep CNN

文献链接：[ImageNet Classification with Deep Convolutional Neural Networks (nips.cc)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

几个引用：<br/>
①上下采样：https://zhuanlan.zhihu.com/p/157352343<br/>
②池化：
https://zhuanlan.zhihu.com/p/453740631 <br/>
③ReLU：
https://zhuanlan.zhihu.com/p/428448728 <br/>

## Abstract

在2010年的 ImageNet 的比赛中，作者训练了一个大型的 CNN 网络去对 1.2 million 个高分辨率的图片进行1000个类别的分类。并在这个比赛中SOTA。这个神经网络有 60 million 个参数和 650000 个神经元，包含了 5 个卷积层，其中的一些还带有最大池化层（ max-pooling layers ），还有三个全连接层（ fully-connected layers ），with 1000-way softmax 。为了使训练更快，作者采用了不饱和的神经元和一个GPU。为了减少在全连接层的过拟合的情况，作者使用了“dropout”的正则化方法。

## 1. Introduction
1. 第一段讨论了机器学习模型性能与数据集大小的关系，以及现实中图像识别问题的解决效果在数据量上面临的问题。
2. 第二段，引入CNN之前，讲了现在的模型即使有很大的数据量，效果也很差及其原因。所以引入了CNN。
3. 第三段，作者讲了即使CNN很棒，但是在这种大容量的数据集上部署仍然是很 expensive 的。但是也有一些利好的条件。
4. 第四段作者开始巴拉巴拉他的贡献了。

- 训练了一个很牛的CNN，在几个数据集上SOTA。
- 开源了高度优化的2D卷积的GPU实现和训练CNN的其他操作 
- 用一些方法来解决了过拟合问题。

5. 最后一段，作者的意思就是，这个模型的瓶颈在于硬件和时间，如果硬件够好，那么它的训练结果就更好。

## 2. The Dataset

1. 第一段，介绍了数据集，就是摘要里的那个。介绍了这个数据集的来源，以及相关的比赛。
2. 第二段，介绍了作者拿设计的模型去参加了比赛的情况，以及比赛的几个评价标准。
3. 第三段，隐晦地说了传说中的 **“端到端”** ？作者讲了 ImageNet 由可变分辨率的图像组成，但是作者的系统需要恒定的输入维度。所以进行了下采样（见引用链接1）。然后用一个方法将图像变化到一个 256 * 256 的矩阵里。除此之外，没有进行任何的其他预处理操作（还有一步 subtracting the mean activity over the training set from each pixel）。

## 3. The Architecture

下面这个图描述了这个模型的架构。包含了五个卷积层和三个全连接层。然后按重要性依次讲了 4 个创新点吧算是。


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/63d45004462047d9be625818a564f6a9~tplv-k3u1fbpfcp-watermark.image?)

### 3.1 ReLU Nonlinearity
