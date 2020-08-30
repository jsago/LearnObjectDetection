## CNN思考之坐标回归

---

> 论文：
>
> [How much position information do convolutional neural networks encode? ]
>
> 《An intriguing failing of convolutional neural networks and the CoordConv solution》
>
> 博客：
>
> [谈谈CNN中的位置和尺度问题](<https://zhuanlan.zhihu.com/p/113443895>)
>
> [为什么我的CNN石乐志？我只是平移了一下图像而已](<https://zhuanlan.zhihu.com/p/37949868>)
>
> [CNN的平移不变性回来了，ImageNet准确度还提升了：Adobe开源新方法，登上ICML](<https://zhuanlan.zhihu.com/p/76030489>)
>
> [CNN是怎么学到图片内的绝对位置信息的?](<https://zhuanlan.zhihu.com/p/99766566>)
>
> [深度学习上演“皇帝的新衣”如何剖析CoordConv?](<http://m.elecfans.com/article/712783.html>)
>
> [卷积神经网络的问题及其解决方案CoordConv](<http://www.elecfans.com/d/711365.html>)
>
> [Uber提出CoordConv：解决普通CNN坐标变换问题](<https://zhuanlan.zhihu.com/p/39919038>)





1. 位置信息是zero-padding透露的