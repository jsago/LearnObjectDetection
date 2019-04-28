

# Git 学习笔记

---

> 参考资料：
>
> [官方中文教程](<https://git-scm.com/book/zh/v2>)
>
> [Git的那些奇技淫巧](<https://mp.weixin.qq.com/s/rk8e77pYkbsm9NEKA7n_sA>)
>
> [git-tips](<https://github.com/521xueweihan/git-tips>)
>
> [自学Git，有哪些书籍或者好的学习资源？](<https://www.zhihu.com/question/38008771/answer/517332296>)

---

[TOC]



## Git是什么？

![123](https://git-scm.com/book/en/v2/images/distributed.png)

![img](https://pic2.zhimg.com/80/v2-3bc9d5f2c49a713c776e69676d7d56c5_hd.jpg)



工作区：就是你在电脑上看到的目录，比如目录下的文件(.git隐藏目录版本库除外)。或者以后需要再新建的目录文件等等都属于工作区范畴。

版本库(Repository)：工作区有一个隐藏目录.git,这个不属于工作区，这是版本库。其中版本库里面存了很多东西，其中最重要的就是stage(暂存区)，还有Git为我们自动创建了第一个分支master,以及指向master的一个指针HEAD。

使用Git提交文件到版本库有两步：
第一步：是使用 `git add` 把文件添加进去，实际上就是把文件添加到暂存区。
第二步：使用`git commit`提交更改，实际上就是把暂存区的所有内容提交到当前分支上。

![å·¥ä½ç®å½ãæå­åºåä"¥å Git ä"åºã](https://git-scm.com/book/en/v2/images/areas.png)



## 设置查看基本信息

```shell
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

当你想针对特定项目使用不同的用户名称与邮件地址时，可以在那个项目目录下运行没有 `--global` 选项的命令来配置

```shell
# 列出git能找到的所有配置
git config --list
# 可以通过`git config <key>`检查每一项的配置
git config user.name
```

```markdown
输出：
user.name=John Doe
user.email=johndoe@example.com
color.status=auto
color.branch=auto
color.interactive=auto
color.diff=auto
```



```shell
# 获取帮助
git help <verb>
git <verb> --help
man git-<verb>

# e.g. 获取config 的帮助
git help config
```

```shell
# 获取版本号
git reflog
```



## 基本命令

### 初始化

```shell
# 初始化，把目录变成git 可管理的仓库
git init
```

### 添加到暂缓区

```shell
# 添加到暂缓区，开始跟踪一个文件
# 或者把已跟踪的文件放到暂存区，（运行了git add之后修改的文件需要再次git add）
# 还能用于合并时把有冲突的文件标记为已解决状态等
# 添加内容到下一次提交中
git add $filename
git add *
```

### 提交注释

```shell
# 提交注释
git commit  # 启动文本编辑器，输入提交说明
git commit -m "提交注释" # -m 将提交信息与命令放在同一行
git commit -a # git 自动把所有已经跟踪过的文件暂存起来一并提交，跳过了git add步骤
git commit -a -m 'added new benchmarks'
git commit --amend # 重新提交，代替上一次提交的结果
```

### 查看更改

```shell
# 查看尚未暂存的文件更新了哪些部分
# 此命令比较的是工作目录中当前文件和暂存区域快照之间的差异， 
# 也就是修改之后还没有暂存起来的变化内容
git diff $filename
# 若要查看已暂存的将要添加到下次提交里的内容
git diff --cached
git diff --staged
```

`git diff` 本身只显示尚未暂存的改动，而不是自上次提交以来所做的所有改动

所以有时候你一下子暂存了所有更新过的文件后，运行 `git diff` 后却什么也没有

### 查看状态

```shell
# 查看git状态

git status
git status -s       # 得到更为紧凑的输出
git status --short   # 同上

# e.g
git status -s
# 输出
 M README             # README 文件在工作区被修改了但是还没有将修改后的文件放入暂存区
MM Rakefile           # Rakefile 在工作区被修改并提交到暂存区后又在工作区中被修改了
A  lib/git.rb         # 新添加到暂存区中的文件前面有 A 标记，修改过的文件前面有 M 标记
M  lib/simplegit.rb   # lib/simplegit.rb 文件被修改了并将修改后的文件放入了暂存区
?? LICENSE.txt        # 新添加的未跟踪文件前面有 ?? 标记
```

### 查看日志

```shell
# 查看历史状态，显示日志
git log
git log –pretty=oneline # 指定使用不同于默认格式的方式展示提交历史
git log -p -2 # -p显示每次提交的内容差异，-2表示仅显示最近两次的提交
git log --stat # 每次提交的简略的统计信息
git log --pretty=format:"%h - %an, %ar : %s" # 定制显示的记录格式
git log --since=2.weeks # 只显示最近两周内的提交
git log -S function_name # 添加或移除了某一个特定函数的引用的提交
```

```shell
$ git log --pretty=format:"%h - %an, %ar : %s"
ca82a6d - Scott Chacon, 6 years ago : changed the version number
085bb3b - Scott Chacon, 6 years ago : removed unnecessary test
a11bef0 - Scott Chacon, 6 years ago : first commit
```

`git log`常用选项

| 选项              |                             说明                             |
| ----------------- | :----------------------------------------------------------: |
| `-p`              |              按补丁格式显示每个更新之间的差异。              |
| `--stat`          |               显示每次更新的文件修改统计信息。               |
| `--shortstat`     |         只显示 --stat 中最后的行数修改添加移除统计。         |
| `--name-only`     |             仅在提交信息后显示已修改的文件清单。             |
| `--name-status`   |               显示新增、修改、删除的文件清单。               |
| `--abbrev-commit` |      仅显示 SHA-1 的前几个字符，而非所有的 40 个字符。       |
| `--relative-date` |       使用较短的相对时间显示（比如，“2 weeks ago”）。        |
| `--graph`         |             显示 ASCII 图形表示的分支合并历史。              |
| `--pretty`        | 使用其他格式显示历史提交信息。可用的选项包括 oneline，short，full，fuller 和 format（后跟指定格式）。 |

`git log`限制输出选项

| 选项                  | 说明                               |
| --------------------- | ---------------------------------- |
| `-(n)`                | 仅显示最近的 n 条提交              |
| `--since`, `--after`  | 仅显示指定时间之后的提交。         |
| `--until`, `--before` | 仅显示指定时间之前的提交。         |
| `--author`            | 仅显示指定作者相关的提交。         |
| `--committer`         | 仅显示指定提交者相关的提交。       |
| `--grep`              | 仅显示含指定关键字的提交           |
| -S                    | 仅显示添加或移除了某个关键字的提交 |



### 版本回退， 取消暂存

```shell
# 版本回退
git reset --hard HEAD^ # 回退到上一个版本
git reset --hard HEAD^^ # 回退到上上个版本
git reset --hard 6fcfc89 # 通过版本号回退
```



```shell
# 取消暂存
git reset HEAD <file>
```



### 移除文件

```shell
# 移除文件
# 先手工删除，然后运行`git rm`记录一下
git rm PROJECTS.md
# 把文件从 Git 仓库中删除（亦即从暂存区域移除），但仍然希望保留在当前工作目录中
git rm --cached README

```

 如果删除之前修改过并且已经放到暂存区域的话，则必须要用强制删除选项 `-f`（译注：即 force 的首字母）



### 移动文件

```shell
# 移动文件
git mv file_from file_to
```

运行 `git mv` 就相当于运行了下面三条命令：

```shell
mv README.md README
git rm README.md
git add README
```



### 撤销修改

```shell
# 撤销修改
git checkout -- CONTRIBUTING.md
```

> 你需要知道 `git checkout -- [file]` 是一个危险的命令，这很重要。 你对那个文件做的任何修改都会消失 - 你只是拷贝了另一个文件来覆盖它。 除非你确实清楚不想要那个文件了，否则不要使用这个命令。

### 打标签

```shell
# 列出标签
git tag
v0.1
v1.3
```

如果只对1.8.5系列感兴趣

```shell
git tag -l 'v1.8.5*'
v1.8.5
v1.8.5-rc0
v1.8.5-rc1
v1.8.5-rc2
v1.8.5-rc3
v1.8.5.1
v1.8.5.2
v1.8.5.3
v1.8.5.4
v1.8.5.5
```

创建标签: 轻量标签和附注标签

```shell
# 附注标签
git tag -a v1.4 -m 'my version 1.4'
# -m 选项指定了一条将会存储在标签中的信息

# 标签信息与对应的提交信息
git show

# 轻量标签
git tag v1.4-lw
```

后期打标签

```shell
#查看提交历史
git log --pretty=oneline

#为对应的提交打标签
git tag -a v1.2 9fceb02
```

共享标签

```shell
# 必须显示的推送标签
git push origin v1.5
git push origin --tags # 推送所有不在远程仓库的标签
```

删除标签

```shell
git tag -d v1.4-lw # -d表示删除
git push origin :refs/tags/v1.4-lw  # 更新，从远程仓库删除该标签
```

检出标签

```shell
git checkout 2.0.0
```



## 远程仓库

### 克隆仓库

```shell
# 克隆远程仓库
git clone https://github.com/libgit2/libgit2  # 克隆远程仓库
git clone https://github.com/libgit2/libgit2 mylibgit #克隆远程仓库并重命名
```

```shell
# 显示需要读写远程仓库使用的git保存的简写与其对应的 URL
git remote -v

# 远程仓库不止一个
cd grit
git remote -v
# 输出
bakkdoor  https://github.com/bakkdoor/grit (fetch)
bakkdoor  https://github.com/bakkdoor/grit (push)
cho45     https://github.com/cho45/grit (fetch)
cho45     https://github.com/cho45/grit (push)
defunkt   https://github.com/defunkt/grit (fetch)
defunkt   https://github.com/defunkt/grit (push)
koke      git://github.com/koke/grit.git (fetch)
koke      git://github.com/koke/grit.git (push)
origin    git@github.com:mojombo/grit.git (fetch)
origin    git@github.com:mojombo/grit.git (push)
```

### 添加远程仓库

```shell
# 添加远程仓库
git remote add <shortname> <url>
git remote add pb https://github.com/paulboone/ticgit
# 现在你可以在命令行中使用字符串 pb 来代替整个 URL。
```

### 拉取数据

```shell
# 从远程仓库拉取没有的数据
git fetch [remote-name]
# 执行完成后，你将会拥有那个远程仓库中所有分支的引用，可以随时合并或查看
# 注意 git fetch 命令会将数据拉取到你的本地仓库 - 它并不会自动合并或修改你当前的工作
```

如果你有一个分支设置为跟踪一个远程分支，可以使用 `git pull`命令来自动的抓取然后合并远程分支到当前分支。 

```shell
# 抓取所有的远程应用
git pull
```



### 推送到远程仓库

```shell
git push [remote-name] [branch-name]
# 将 master 分支推送到 origin 服务器
git push origin master
```

>  当你和其他人在同一时间克隆，他们先推送到上游然后你再推送到上游，你的推送就会毫无疑问地被拒绝。 你必须先将他们的工作拉取下来并将其合并进你的工作后才能推送



### 查看远程仓库

```shell
 git remote show [remote-name]
 git remote show origin
```



### 远程仓库的移除与重命名

```shell
# 重命名
git remote rename pb paul
# 移除一个远程仓库
git remote rm paul
```



## 技巧

### 忽略文件`.gitignore`

1. 如何设置`.gitignore`文件，忽略文件

> `GitHub` 有一个十分详细的针对数十种项目及语言的 `.gitignore` 文件列表，你可以在 <https://github.com/github/gitignore> 找到它

忽略文件，设置`.gitignore`

```shell
$ cat .gitignore
*.[oa]
*~
```

第一行：忽略所有以.o或.a结束的文件

第二行：忽略所有以波浪符结尾的文件



文件 `.gitignore` 的格式规范如下：

- 所有空行或者以 `＃` 开头的行都会被 Git 忽略。
- 可以使用标准的 glob 模式匹配。
- 匹配模式可以以（`/`）开头防止递归。
- 匹配模式可以以（`/`）结尾指定目录。
- 要忽略指定模式以外的文件或目录，可以在模式前加上惊叹号（`!`）取反。



### git 别名

```shell
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```

e.g1

```shell
git config --global alias.unstage 'reset HEAD --'
# 下面两个命令等效
git unstage fileA
git reset HEAD -- fileA
```

e.g2

```shell
git config --global alias.last 'log -1 HEAD'
# 轻松看最后一次提交
git last
```

