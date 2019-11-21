# Attention

---

> 论文：《An Empirical Study of Spatial Attention Mechanisms in Deep Networks》
>
> ​             《Non-local Neural Networks》
>
> 书籍：《神经网络与深度学习》第八章    邱锡鹏  
>
> 参考博客：
>
> [计算机视觉中的注意力机制](<https://zhuanlan.zhihu.com/p/56501461>)
>
> [计算机视觉中attention机制的理解](<https://zhuanlan.zhihu.com/p/61440116>)
>
> [微软亚研：对深度神经网络中空间注意力机制的经验性研究](<https://www.jiqizhixin.com/articles/2019-04-15-12>)
>
> [nlp中的Attention注意力机制+Transformer详解](<https://zhuanlan.zhihu.com/p/53682800>)
>
> [细讲 | Attention Is All You Need](<https://mp.weixin.qq.com/s/RLxWevVWHXgX-UcoxDS70w>)
>
> [深度学习中的注意力模型（2017版）](<https://zhuanlan.zhihu.com/p/37601161>)
>
> [【AI不惑境】计算机视觉中注意力机制原理及其模型发展和应用](<https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649034948&idx=1&sn=f02b7d42d72cadfa50ab34cfeffb36af&chksm=8712aeb9b06527af4d06105d95503d6f36bd287293e2e8b76e6647858e109a201c1efe763599&token=1296655195&lang=zh_CN#rd>)
>
> [Attention算法调研（视觉应用概况）](<https://zhuanlan.zhihu.com/p/52925608>)
>
> [Attention算法调研(五) —— 视觉应用中的Self Attention](<https://zhuanlan.zhihu.com/p/53155423>)
>
> [用自注意力增强卷积：这是新老两代神经网络的对话（附实现）](<https://zhuanlan.zhihu.com/p/63910019>)
>
> 

## Non-local Network

[【论文阅读】Non-local Neural Networks](<https://blog.csdn.net/u013859301/article/details/80167758>)

[Non-local neural networks](<https://zhuanlan.zhihu.com/p/33345791>)

[Non-local Neural Networks及自注意力机制思考](<https://zhuanlan.zhihu.com/p/53010734>)

[当Non-local遇见SENet，微软亚研提出更高效的全局上下文网络](<https://zhuanlan.zhihu.com/p/64863345>)





![img](assets/v2-96ea13e5bb836e197959da43126d4c3c_hd.jpg)



**将位置或者空间注意力机制应用到了所有通道的每张特征图对应位置上，本质就是输出的每个位置值都是其他所有位置的加权平均值，通过softmax操作可以进一步突出共性**

## Attention 机制计算流程

![img](assets/v2-54fe529ded98721f35277a5bfa79febc_hd.jpg)



![img](assets/v2-83b682055e93f41cd0b2ff095809dd9e_b.jpg)



**Attention机制的实质其实就是一个寻址（addressing）的过程**，如上图所示：给定一个和任务相关的查询**Query**向量 **q**，通过计算与**Key**的注意力分布并附加在**Value**上，从而计算**Attention Value**

![img](assets/v2-76cac5c196e43afc8338712b6a41d491_hd.jpg)



注意力机制可以分为三步：一是信息输入；二是计算注意力分布α；三是根据注意力分布α 来计算输入信息的加权平均。**

![1567758213944](assets/1567758213944.png)

两种软性注意力机制的模式

![preview](assets/v2-aa371755dc73b7137149b8d2905fc4ba_r.jpg)

键值对模式

![1567758377149](assets/1567758377149.png)





## 四个可能的注意力因素





![1567478838846](目标检测论文/assets/1567478838846.png)



## 注意力机制的种类

### 硬性注意力机制和软性注意力机制

![preview](目标检测论文/assets/v2-b968a4c2d5533d3154ecf12ef27c11d6_r.jpg)

### Self-Attention

![img](assets/v2-fcc2df696966a9c6700d1476690cff9f_hd.jpg)

### Transformer

![img](assets/v2-7f8b460cd617fedc822064c4230302b0_hd.jpg)



## Spatial Attention Mechanisms

![1567482869845](目标检测论文/assets/1567482869845.png)

## 关键论文

### 1. 《Look Closer to See Better：Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition》

![preview](目标检测论文/assets/v2-3e8035b21f04058c6fbbcbe6eb9224b2_r.jpg)



Attention Proposal Sub-Network（APN）。这个 APN 结构是从整个图片（full-image）出发，迭代式地生成子区域，并且对这些子区域进行必要的预测，并将子区域所得到的预测结果进行必要的整合，从而得到整张图片的分类预测概率 

### 2. 《Multiple Granularity Descriptors for Fine-grained Categorization》

### 3. 《Recurrent Models of Visual Attention》