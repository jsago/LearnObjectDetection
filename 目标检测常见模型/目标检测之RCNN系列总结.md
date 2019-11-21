# RCNN系列

---



## Faster-RCNN

## Cascade-RCNN

> 参考资料：
>
> [目标检测论文阅读：Cascade R-CNN: Delving into High Quality Object Detection](<https://blog.csdn.net/qq_21949357/article/details/80046867>)
>
> [目标检测-Cascade R-CNN-论文笔记](<https://arleyzhang.github.io/articles/1c9cf9be/>)
>
> [Cascade R-CNN 详细解读](<https://zhuanlan.zhihu.com/p/42553957>)



针对的是检测问题中的IoU阈值选取问题，众所周知，阈值选取越高就越容易得到高质量的样本，但是一味选取高阈值会引发两个问题：

- 样本减少引发的过拟合
- 在train和inference使用不一样的阈值很容易导致mismatch

能否有一种方式既可以用较高的IoU阈值训练检测器，又能够保证正样本的diversity足够丰富？基于以上分析，下面我们详细论述上作者所提出的Cascade R-CNN，其核心思想就是‘分而治之’。



![img](assets/16590019d4683cf3)



### mismatch问题

training阶段和inference阶段，bbox回归器的输入分布是不一样的，training阶段的输入proposals质量更高(被采样过，IoU>threshold)，inference阶段的输入proposals质量相对较差（没有被采样过，可能包括很多IoU<threshold的），这就是论文中提到**mismatch**问题，这个问题是固有存在的，通常threshold取0.5时，mismatch问题还不会很严重。