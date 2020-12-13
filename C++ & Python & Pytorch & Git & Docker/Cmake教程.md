# Cmake简易教程

---

> 参考：
>
> [cmake使用教程（一）-起步](<https://juejin.im/post/6844903557183832078>)



---



最基本的cmake:

```cmake
cmake_minimum_required (VERSION 2.6)
project (Tutorial)
add_executable(Tutorial tutorial.cxx)
```



> 系统指令是不区分大小写的，但是变量和字符串是区分大小写的



编译：生成了三个文件`CMakeCache.txt`、`Makefile`、`cmake_install.cmake`和一个文件夹`CmakeFiles`

```cmake
cmake .
```



执行：

```cmake
make
```

