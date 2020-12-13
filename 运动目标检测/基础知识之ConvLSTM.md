# ConvLSTM

---

> 论文：《Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting》
>
> 博客：
>
> [人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)
>
> [[理解 LSTM(Long Short-Term Memory, LSTM) 网络](https://www.cnblogs.com/wangduo/p/6773601.html)]



---

## LSTM简介

原始RNN：

![img](assets/v2-f716c816d46792b867a6815c278f11cb_720w.jpg)



LSTM：简介

![img](assets/v2-e4f9851cad426dfe4ab1c76209546827_720w.jpg)



LSTM：详细介绍

![img](assets/v2-556c74f0e025a47fea05dc0f76ea775d_720w.jpg)



![1600842820907](assets/1600842820907.png)



![img](assets/42741-b9a16a53d58ca2b9.png)



**这里面的关于权重W的操作，跟全连接类似，因此这种lstm又可以叫FC-LSTM。这种结构非常擅长处理时序信息，也能处理空间信息**

## ConvLSTM

![preview](assets/v2-84b2e0f33068cfadd03875e1a1ef2253_r.jpg)