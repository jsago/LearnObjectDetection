# SSD

---

> 参考资料：
>
> [SSD原理解读-从入门到精通](<https://blog.csdn.net/qianqing13579/article/details/82106664>)
>
> [目标检测-SSD-Single Shot MultiBox Detector-论文笔记](<https://arleyzhang.github.io/articles/786f1ca3/>)
>
> [论文阅读：SSD: Single Shot MultiBox Detector](<https://blog.csdn.net/u010167269/article/details/52563573>)
>
> [目标检测|SSD原理与实现](<https://zhuanlan.zhihu.com/p/33544892>)
>
> [**a-PyTorch-Tutorial-to-Object-Detection**](<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection>)
>
> 
>
> 源码：
>
> [ssd.pytorch](<https://github.com/amdegroot/ssd.pytorch>)

---

## 基本原理

### 模型框架

![1521700215199](https://arleyzhang.github.io/articles/786f1ca3/1521700215199.png)

![img](https://arleyzhang.github.io/articles/786f1ca3/1521723003643.png)



采用VGG16作为base model

#### 基本特征

1. 多尺度

比较大的特征图来用来检测相对较小的目标，而小的特征图负责检测大目标

2. 采用卷积进行检测

与Yolo最后采用全连接层不同，SSD直接采用卷积对不同的特征图来进行提取检测结果

3. 设置先验框



SSD是一个纯卷积神经网络(CNN)，可以分成三个部分：

+ 基本卷积：来自于现有的图像分类体系结构，该体系结构将提供较低级别的特征映射。
+ 辅助卷积：在基本网络之上添加的辅助卷积将提供更高级别的特征映射。
+ 预测卷积：它将定位和识别这些特征映射中的对象



FC层与Conv层的转化



#### Prior

SSD300的priord

不同层会采用不同比例的priors

另外，不同层都有一个额外的1：1的先验框，尺度是当前以及子feature map尺度的几何平均？

| Feature Map From | Feature Map Dimensions | Prior Scale | Aspect Ratios                            | Number of Priors per Position | Total Number of Priors on this Feature Map |
| ---------------- | ---------------------- | ----------- | ---------------------------------------- | ----------------------------- | ------------------------------------------ |
| `conv4_3`        | 38, 38                 | 0.1         | 1:1, 2:1, 1:2 + an extra prior           | 4                             | 5776                                       |
| `conv7`          | 19, 19                 | 0.2         | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6                             | 2166                                       |
| `conv8_2`        | 10, 10                 | 0.375       | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6                             | 600                                        |
| `conv9_2`        | 5, 5                   | 0.55        | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6                             | 150                                        |
| `conv10_2`       | 3, 3                   | 0.725       | 1:1, 2:1, 1:2 + an extra prior           | 4                             | 36                                         |
| `conv11_2`       | 1, 1                   | 0.9         | 1:1, 2:1, 1:2 + an extra prior           | 4                             | 4                                          |
| **Grand Total**  | –                      | –           | –                                        | –                             | **8732 priors**                            |

priors长宽的确定方法：

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/wh1.jpg)

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/wh2.jpg)

例如，在Conv9_2层

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/priors1.jpg)





#### predict Conv预测卷积模块

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv1.jpg)

Conv9_2:   6表示6个先验框

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv2.jpg)

位置预测：

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv3.jpg)

类别预测：

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv4.jpg)



![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/reshaping1.jpg)

则，所有feature map上的预测聚合：

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/reshaping2.jpg)



### 匹配策略

在SSD中，通俗的说就是先产生一些预选的default box（类似于anchor box），然后标签是 ground truth box，预测的是bounding box，现在有三种框，从default box到ground truth有个变换关系，从default box到prediction bounding box有一个变换关系，如果这两个变换关系是相同的，那么就会导致 prediction bounding box 与 ground truth重合

![1522219622636](https://arleyzhang.github.io/articles/786f1ca3/1522219622636.png)



1. 找出8732个先验和N个地面真值对象之间的重叠部分。这将是一个大小为8732 * n的张量，
2. 将8732个先验中的每一个匹配到它与之有最大重叠的对象。
3. 如果先验与Jaccard重叠小于0.5的对象匹配，则不能说先验“包含”该对象，因此是负匹配。考虑到我们有成千上万的先验，大多数先验测试结果都是阴性的。
4. 另一方面，一些先验实际上会与一个对象显著重叠(大于0.5)，并且可以说“包含”了该对象。这些是积极的匹配。
5. 现在，我们已经将8732个先验中的每一个都匹配到一个基本事实，实际上，我们也将相应的8732个预测匹配到一个基本事实。



Example: 假设只有7个priors

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/matching1.PNG)

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/matching2.jpg)



### 损失函数

Multibox loss

![img](https://images2017.cnblogs.com/blog/1067530/201708/1067530-20170811175226976-860447034.png)

SSD包含三部分的loss：前景分类的loss、背景分类的loss、位置回归的loss

大部分预测边框不包含对象

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/confloss.jpg)

### 边框回归

预测偏置

![img](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/ecs2.PNG)



### 预测结果后处理

对于8732个先验框预测结果（坐标偏置），转换为边框坐标

+ 为8732个框中的每一个提取这个类的分数。
+ 排除不符合此分数特定阈值的框。
+ 其余的(未删除的)框是这个特定对象类的候选框。

如果此时绘制候选框，同一个目标会出现许多重叠的候选框

采用NMS进行处理，保留最大score的框，抑制IOU超过阈值的框



## SSD的优势

1. 多尺度：采用6个不同的特征图检测不同尺度的目标
2. 设置了多种宽高比的default box：
3. 数据增强：放大/缩小+随机Crop

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180825141941407?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5xaW5nMTM1Nzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180825142837948?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpYW5xaW5nMTM1Nzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## SSD的缺点

SSD对小目标的检测效果一般

对SSD的改进可以从下面几个方面考虑： 
1. 增大输入尺寸 
2. 使用更低的特征图做检测 
3. 设置default box的大小，让default box能够更好的匹配实际的有效感受野



## SSD 与MTCNN

## SSD 源码解读

### ssd.pytorch

1. 计算mAP是否使用VOC07的方法?

VOC2007的mAP评价方法是采用取11个点的平均值的方法：取大于t的recall的所有结果中最大的precision并对11个点取平均

```python
def safehat_ap(rec, prec, use_07_metric=True):
    """ ap = safehat_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    SafeHat 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        # 倒序遍历：与后一项相比，取大值
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```

