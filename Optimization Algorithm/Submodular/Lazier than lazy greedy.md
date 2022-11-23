Lazier than lazy greedy。作者提出来一个随机贪心策略。

# 2022.11.23

## Algorithm
浅读了一下，这个"随机贪心策略"算法的步骤如下：


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c03b8432ddbd42e2806cf7363dbda941~tplv-k3u1fbpfcp-watermark.image?)

先初始化一个空集A，然后对V-A进行采样R，采样的大小见下图。然后从R中选一个使得Δ(a|A)最大的元素a，加入到A中。循环k次。


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1118cd39d3e54b77a1bb919de793965f~tplv-k3u1fbpfcp-watermark.image?)

## Experiment

这个验证思路还蛮不错的。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/350e867d7d60405c8c236476f3278449~tplv-k3u1fbpfcp-watermark.image?)
