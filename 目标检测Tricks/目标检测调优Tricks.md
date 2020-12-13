# 目标检测调优tricks

---

> 参考资料：
>
> [**检测模型中的Bells and wisthles**](<https://cloud.tencent.com/developer/article/1109497>)
>
> [目标检测调优技巧](<http://www.pianshen.com/article/2781276525/>)
>
> [Bag of Freebies for Training Object Detection Neural Networks学习笔记](<https://www.jianshu.com/p/003c5b6477ba>)
>
> [百度视觉团队斩获 ECCV Google AI 目标检测竞赛冠军，获奖方案全解读 | ECCV 2018](<https://www.leiphone.com/news/201809/6T23aH3sevzHbeJR.html>)
>
> [Kaggle新手银牌（21st）：Airbus Ship Detection 卫星图像分割检测](<https://zhuanlan.zhihu.com/p/48381892>)
>
> 论文:
>
> 《Bag of Freebies for Training Object Detection Neural Networks》

---

## 测试阶段 inference ”tricks”

多尺度测试、水平翻转、窗口微调与多窗口投票、多模型融合、NMS阈值调整、多模型融合

单图多scale检测投票



### 引入多尺度测试Multi Scale Testing

RefineDet:  test/lib/fast_rcnn

<https://github.com/sfzhang15/RefineDet/blob/master/test/lib/fast_rcnn/test.py>



Maxout:

### soft-NMS

通常为贪心式方法：greedy nms

通常的做法是将检测框按得分排序，然后保留得分最高的框，同时删除与该框重叠面积大于一定比例的其它框。



soft nms

[一行代码改进NMS](<https://blog.csdn.net/shuzfan/article/details/71036040>)

 [《Improving Object Detection With One Line of Code》](http://cn.arxiv.org/abs/1704.04503)

Github链接： <https://github.com/bharatsingh430/soft-nms>



**不要粗鲁地删除所有IOU大于阈值的框，而是降低其置信度。**



伪代码如下：

![float](https://img-blog.csdn.net/20170430181251698?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2h1emZhbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



文中有两种方式：

![1557147329570](assets/1557147329570.png)





参考代码：

```python
def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503     https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])
```







### 预测框微调/投票法

不同的训练策略，不同的 epoch 预测的结果，使用 NMS 来融合，或者sof-tnms



需要调整的参数：

-  box voting 的阈值，
- 不同的输入中**这个框至少出现了几次来**允许它输出，
- 得分的阈值，一个目标框的得分低于这个阈值的时候，就删掉这个目标框。

![img](https://img-blog.csdn.net/20171012214545776?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2ZlaTEwMQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### Ensemble集成方法

使用差异较大的模型融合，融合策略一般采用NMS



[示例代码1](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/ensemble_predictions_with_weighted_method.py)

模型融合：

对于每个模型，百度视觉团队在 NMS 后预测边界框。来自不同模型的预测框则使用一个改进版的 NMS 进行合并，具体如下：

- 给每个模型一个 0～1 之间的标量权重。所有的权重总和为 1；
- 从每个模型得到边界框的置信分数乘以它对应的权重；
- 合并从所有模型得到的预测框并使用 NMS，除此之外百度采用不同模型的分数叠加的方式代替只保留最高分模型

![ç¾åº¦è§è§å¢éæ©è· ECCV Google AI ç®æ æ£æµç"èµå åï¼è·å¥æ¹æ¡å¨è§£è¯" |  ECCV 2018](assets/5b9a293342a24.jpg)

![ç¾åº¦è§è§å¢éæ©è· ECCV Google AI ç®æ æ£æµç"èµå åï¼è·å¥æ¹æ¡å¨è§£è¯" |  ECCV 2018](assets/5b9a2938206d2.jpg)



![1560863784319](assets/1560863784319.png)

### 图像增强+NMS

水平翻转或者旋转等等