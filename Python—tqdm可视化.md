# Tqdm进度条可视化

---

> 参考资料
>
> [tqdm介绍及常用方法](<https://blog.csdn.net/zkp_987/article/details/81748098>)

---

Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。 

## 方法一;  tqdm(list)

可传入任意list

```python
from tqdm import tqdm

for i in tqdm (range(1000)):
    # do
    pass
```



也可传入string的数组

```python
for char in tqdm(["a","b","c","d"]):
    # do
    pass
```



## 方法二：trange

```python
from tqdm import trange

for i in trange(100):
    # do
    pass
```



## 方法三：手动方法

for 循环外部初始化

```python
bar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    pbar.set_description("Processing %s" % char)
```

