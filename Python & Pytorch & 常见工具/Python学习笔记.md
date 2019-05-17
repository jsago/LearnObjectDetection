## 1. Python中的5种下划线

[Python中下划线的5种含义](<https://blog.csdn.net/tcx1992/article/details/80105645>)



## 2. Python中文编码

```python
# -*- coding: UTF-8 -*-
```



## 3. zip()

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```

## 4. format与%

[基础_格式化输出（%用法和format用法）](https://www.cnblogs.com/fat39/p/7159881.html)

[python中使用%与.format格式化文本](https://www.cnblogs.com/engeng/p/6605936.html)

