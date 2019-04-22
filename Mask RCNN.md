# Mask RCNN

---

>参考资料
>
>[令人拍案称奇的Mask RCNN](<https://zhuanlan.zhihu.com/p/37998710>)
>
>[知乎：如何评价 Kaiming He 最新的 Mask R-CNN?](<https://www.zhihu.com/question/57403701>)
>
>论文：
>
>《[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)》
>
>《Mask R-CNN》

---

算法需要解决四个问题，其一是图像分类 (image classification)，主要回答是什么的问题，其二是定位 (localization)，主要回答在哪里的问题，其三是mask classification，即对图像中每个像素做分类，进一步回答了在哪里的问题，其四是关键点定位，在进一步回答在哪里的同时也包含了更多是什么的信息。



## 基本架构

ResNet-FPN+Fast RCNN+mask

主要的改进点：

1. 基础网络的增强：FPN
2. 分割loss的改进：由原来的 FCIS 的 基于单像素softmax的多项式交叉熵变为了基于单像素sigmod二值交叉熵
3. ROI Align层的加入
4. 添加并列的FCN层，即Mask层



### FPN

引入FPN是对主干网络的主要扩展

![img](https://image.jiqizhixin.com/uploads/editor/35f55332-e651-42a4-8959-fc93dcad8003/1521687745095.jpg)

FPN产生了特征金字塔 $[P2,P3,P4,P5,P6]​$，而并非只是一个feature map。金字塔经过RPN之后会产生很多region proposal

Top-Down + Bottom-Up

FPN结构中包括自下而上，自上而下和横向连接三个部分

![img](https://pic4.zhimg.com/80/v2-fc500b77472298d7dacdd303f509c68b_hd.jpg)

1. 自下而上

特征提取的过程

2. 自上而下

从最高层开始进行上采样，采用最近邻上采样

3. 横向连接



### Mask分支

与FasterRCNN相比，Mask-RCNN多了一个分支；Mask-RCNN将RCNN拓展到语义分割领域

Mask-RCNN的实现是FCN网络，掩码分支实际就是一个卷积网络，选择ROI分类器的正样本作为输入，生成对应的掩码



网络架构：

1. 骨干网络ResNet-FPN，用于特征提取，另外，ResNet还可以是：ResNet-50,ResNet-101,ResNeXt-50,ResNeXt-101；
2. 头部网络，包括边界框识别（分类和回归）+mask预测。头部结构见下图



Mask RCNN定义多任务损失：$L=L_{cls}+L_{box}+L_{mask}​$

