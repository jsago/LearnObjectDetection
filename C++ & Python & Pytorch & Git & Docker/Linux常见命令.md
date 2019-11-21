# Linux常见命令

---



1. 查看cuda和cudnn版本

```shell
# cuda 版本 
cat /usr/local/cuda/version.txt

# cudnn 版本 
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```



2. ls查看命令

```shell
# 显示前3行和后3行数据
ls -l|head -n 3
ls -l|tail -n 3
```

\# 查看当前目录下的文件数量（不包含子目录中的文件）

```
ls -l|grep "^-"| wc -l
```

\# 查看当前目录下的文件数量（包含子目录中的文件） 注意：R，代表子目录

```
ls -lR|grep "^-"| wc -l
```

\# 查看当前目录下的文件夹目录个数（不包含子目录中的目录），同上述理，如果需要查看子目录的，加上R

```
ls -l|grep "^d"| wc -l
```

\# 查询当前路径下的指定前缀名的目录下的所有文件数量
\# 例如：统计所有以“20161124”开头的目录下的全部文件数量

```
ls -lR 20161124*/|grep "^-"| wc -l
```