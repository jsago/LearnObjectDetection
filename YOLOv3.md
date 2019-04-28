# YOLOv3

---

> 参考资料：
>
> [官网](<https://pjreddie.com/darknet/yolo/>)
>
> [论文 - YOLO v3：写得不错！！](<https://xmfbit.github.io/2018/04/01/paper-yolov3/>)
>
> [目标检测网络之 YOLOv3](https://www.cnblogs.com/makefile/p/YOLOv3.html)
>
> [YOLOv3：An Incremental Improvement全文翻译](<https://zhuanlan.zhihu.com/p/34945787>)
>
> [如何评价最新的YOLOv3？](<https://www.zhihu.com/question/269909535?rf=269938844>)
>
> [从零开始PyTorch项目：YOLO v3目标检测实现](<https://www.jiqizhixin.com/articles/2018-04-23-3>)
>
> [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 1](<https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/>)
>
> [进击的YOLOv3，目标检测网络的巅峰之作 | 内附实景大片](<https://www.jiqizhixin.com/articles/2018-05-14-4>)
>
> [yolo检测算法系列介绍与yolo v3代码阅读（一）](<https://zhuanlan.zhihu.com/p/41313896>)
>
> 
>
> 源码地址：
>
> <https://github.com/ayooshkathuria/pytorch-yolo-v3>
>
> <https://github.com/eriklindernoren/PyTorch-YOLOv3>
>
> <https://github.com/pjreddie/darknet>
>
> 
>
> 论文：
>
> You Only Look Once: Unified, Real-Time Object Detection
>
> YOLO9000: Better, Faster, Stronger
>
> YOLOv3: An Incremental Improvement

---

## 原理

### 基本思想

#### YOLO

每个格子预测B个bounding box及其置信度(confidence score)，以及C个类别概率

bbox信息(x,y,w,h)为物体的中心位置相对格子位置的偏移及宽度和高度,均被归一化

置信度反映是否包含物体以及包含物体情况下位置的准确

​          $$其中Pr(Object) \times IOU^{truth}_{pred}, 其中Pr(Object)\in\{0,1\}$$



![img](https://image.jiqizhixin.com/uploads/editor/39645335-6ece-4d04-88e4-5782a820cef1/1524466096665.jpg)

Loss函数的计算

![yolo-loss](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181317188-1434000633.png)

![preview](https://pic4.zhimg.com/v2-51c1f881a4edc56140921b08deb1212b_r.jpg)



#### YOLOv2

+ Batch Normalization：取消了定位层后的dropout，在卷积层全部使用batch normlization

+ 高分辨率分辨器：以大尺寸448*448微调原始分类网络

+ Anchor box: v2中移除全连接层，预测bbx的偏移; 对faster rcnn手选先验框的方法做出了改进，采用k-means聚类产生合适的候选框，聚类距离公式如下: 

  $D\text{(box,centroid)} = 1 − IOU\text{(box,centroid)}$

  

+ 细粒度特征： pass through layer

+ multi-scale training

+ 提出darknet-19如图

![Darknet-19-arch](https://images2018.cnblogs.com/blog/606386/201803/606386-20180324181344634-594145493.png)

#### YOLO v3

改进之处：

+ 多尺度预测FPN
+ 更好的基础分类网络Darknet53

下文作详细介绍

### 评价指标

[目标检测中的AP如何计算？](<https://www.zhihu.com/question/41540197>)

什么是AP和mAP？

在detection中，我们认为当预测的bounding box和ground truth的IoU大于某个阈值（如取为0.5）时，认为是一个True Positive。如果小于这个阈值，就是一个False Positive



在检测的任务中，如果取不同的阈值，就可以绘制PR曲线，曲线下的面积就是AP值

COCO中使用了`0.5:0.05:0.95`十个离散点近似计算

detection中常常需要同时检测图像中多个类别的物体，我们将不同类别的AP求平均，就是`mAP`



如果我们只看某个固定的阈值，如0.50.5，计算所有类别的平均AP，那么就用AP50AP50来表示。所以YOLO v3单拿出来AP50AP50说事，是为了证明虽然我的bounding box不如你RetinaNet那么精准（IoU相对较小），但是如果你对框框的位置不是那么敏感（0.5的阈值很多时候够用了），那么我是可以做到比你更好更快的

对于AP50，



### 边框回归

[边框回归(Bounding Box Regression)详解](<https://blog.csdn.net/zijin0802034/article/details/77685438/>)

YOLO预测偏移并不奏效，我们采用的方法不是预测偏移量，而是预测相对于grid cell位置的位置坐标

网络预测四个值：$t_x，t_y，t_w，t_h​$。我们知道，YOLO网络最后输出是一个M×M的feature map，对应于M×M个cell。如果某个cell距离image的top left corner距离为$(c_x,c_y)(​$（也就是cell的坐标），那么该cell内的bounding box的位置和形状参数为图中公式所示

![bounding boxçåå½](https://xmfbit.github.io/img/paper=yolov3-bbox-regression.png)



> PS：这里有一个问题，不管FasterRCNN还是YOLO，都不是直接回归bounding box的长宽（就像这样：$b_w=p_wt′_wb_w=p_wt_w′​$），而是要做一个对数变换，实际预测的是log(⋅)。这里小小解释一下。



仍使用k-means聚类来确定边界框的先验，我们只是选择了9个聚类（clusters）和3个尺度（scales），然后在整个尺度上均匀分割聚类。在COCO数据集上，9个聚类是：（10×13）;（16×30）;（33×23）;（30×61）;（62×45）; （59×119）; （116×90）; （156×198）; （373×326）

K为几，每张图片就聚几类，加上3个尺度， 共3k个框

![1556184330249](assets/1556184330249.png)



### 分类预测

不用softmax做分类了，而是使用独立的logisitc做二分类。这种方法的好处是可以处理重叠的多标签问题，如Open Image Dataset。在其中，会出现诸如`Woman`和`Person`这样的重叠标签

不使用softmax, 主要有以下考虑:

+ Softmax使得每个框分配一个类别（score最大的一个），而对于`Open Images`这种数据集，目标可能有重叠的类别标签，因此Softmax不适用于多标签分类
+ Softmax可被独立的多个logistic分类器替代，且准确率不会下降。
  分类损失采用binary cross-entropy loss.





### 多尺度预测

v3在3个不同的尺度上做预测。在COCO上，我们每个尺度都预测3个框框，所以一共是9个聚类中心。所以输出的feature map的大小是$N\times N\times [3\times (4+1+80)]$

当输入图像大小是 416 x 416 时，我们在尺度 13 x 13、26 x 26 和 52 x 52 上执行检测

![img](https://image.jiqizhixin.com/uploads/editor/3e76ab75-e9af-40cb-b8eb-2ac8cfc1dad0/1524466097727.jpg)

### 输出处理

对于大小为 416 x 416 的图像，YOLO 预测 ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10647 个边界框；但是，我们的示例中只有一个对象——一只狗。

如何减少预测框？

+ 目标置信度阈值：首先，我们根据它们的 objectness 分数过滤边界框。
+ 非极大值抑制：非极大值抑制（NMS）可解决对同一个图像的多次检测的问题。



### 基础网络

DarkNet53

YOLO v3网络结构

![YOLOv3-arch](https://images2018.cnblogs.com/blog/606386/201803/606386-20180327004340505-1572852891.png)

### 补充

**Anchor box** x，y **offset predictions**。我们尝试使用正常anchor box预测机制，这里你使用线性激活来预测x，y offset作为box的宽度或高度的倍数。我们发现这种方法降低了模型的稳定性，并且效果不佳。

**Linear** x，y **predictions instead of logistic**。我们尝试使用线性激活来直接预测x，y offeset 而不是逻辑激活。这导致mAP下降了几个点。

**Focal loss**。我们尝试使用focal loss。它使得mAp降低了2个点。YOLOv3对focal loss解决的问题可能已经很强大，因为它具有单独的对象预测和条件类别预测。 因此，对于大多数例子来说，类别预测没有损失？ 或者其他的东西？ 我们并不完全确定。

**Dual IOU thresholds and truth assignment** 。Faster R-CNN在训练期间使用两个IOU阈值。如果一个预测与ground truth重叠达到0.7，它就像是一个正样本，如果达到0.3-0.7，它被忽略，如果小于0.3，这是一个负样本的例子。我们尝试了类似的策略，但无法取得好成绩。



## 源码解读

GitHub项目名称：`PyTorch-YOLOv3`

### 损失函数

```python
loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
loss_conf_obj = self.bce_loss(pred_conf[obj_mask],\
                              tconf[obj_mask])
loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask],\
                                tconf[noobj_mask])
loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale *loss_conf_noobj
loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
# 所有损失之和
total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
```



### obj_mask、noobj_mask是啥？



### yolo三个layer的预测值都是啥？



## 从零开始实现YOLO v3

