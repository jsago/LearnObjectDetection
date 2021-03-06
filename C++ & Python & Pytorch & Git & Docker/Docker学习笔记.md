# Docker笔记

---

> [菜鸟教程](<http://www.runoob.com/docker/docker-tutorial.html>)
>
> [官方API](<https://docs.docker.com/develop/sdk/>)
>
> [Windows上做Python开发太痛苦？Docker了解一下](<https://zhuanlan.zhihu.com/p/50864774>)
>
> [欢迎来到docker用户指南](<https://www.widuu.com/chinese_docker/userguide/index.html>)
>
> [docker docs](<https://docs.docker.com/get-started/>)
>
> [Docker 容器从入门到入魔](<https://zhuanlan.zhihu.com/p/45610616>)
>
> [【 全干货 】5 分钟带你看懂 Docker ！](<https://zhuanlan.zhihu.com/p/30713987>)

---



## 基本概念

Docker 使用客户端-服务器 (C/S) 架构模式，使用远程API来管理和创建Docker容器

| Docker | 面向对象 |
| :----- | :------- |
| 容器   | 对象     |
| 镜像   | 类       |

| Docker 镜像(Images)    | Docker 镜像是用于创建 Docker 容器的模板。                    |
| ---------------------- | ------------------------------------------------------------ |
| Docker 容器(Container) | 容器是独立运行的一个或一组应用。                             |
| Docker 客户端(Client)  | Docker 客户端通过命令行或者其他工具使用 Docker API (<https://docs.docker.com/reference/api/docker_remote_api>) 与 Docker 的守护进程通信。 |
| Docker 主机(Host)      | 一个物理或者虚拟的机器用于执行 Docker 守护进程和容器。       |
| Docker 仓库(Registry)  | Docker 仓库用来保存镜像，可以理解为代码控制中的代码仓库。Docker Hub([https://hub.docker.com](https://hub.docker.com/)) 提供了庞大的镜像集合供使用。 |
| Docker Machine         | Docker Machine是一个简化Docker安装的命令行工具，通过一个简单的命令行即可在相应的平台上安装Docker，比如VirtualBox、 Digital Ocean、Microsoft Azure。 |

![img](http://www.runoob.com/wp-content/uploads/2016/04/576507-docker1.png)



## docker容器使用

```bash
# 查看docker所有命令
docker

# docker command --help

```

