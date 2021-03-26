# 并发多线程

---

> [C++11多线程并发基础入门教程](https://zhuanlan.zhihu.com/p/194198073?utm_source=com.google.android.gm)
>
> [C++11多线程-条件变量(std::condition_variable)](https://www.bbsmax.com/A/KE5Q11ly5L/)
>
> [Wait/Notify通知机制解析](https://blog.csdn.net/wthfeng/article/details/78762343)
>
> [C++并发编程](https://www.zhihu.com/column/c_1307735602880331776)
>
> [cpp-concurrency-in-action-2ed](https://downdemo.gitbook.io/cpp-concurrency-in-action-2ed/)
>
> 《C++并发编程》

---

## 基本操作

### join/detach

join: 等待子线程结束，回到主线程

整个过程就相当于：你在处理某件事情（你是主线程），中途你让老王帮你办一个任务（与你同时执行）（创建线程1，该线程取名老王），又叫老李帮你办一件任务（创建线程2，该线程取名老李），现在你的一部分工作做完了，剩下的工作得用到他们的处理结果，那就调用"老王.join()"与"老李.join()"，至此你就需要等待（主线程阻塞），等他们把任务做完（子线程运行结束），你就可以继续你手头的工作了（主线程不再阻塞）。

```c++
#include<iostream>
#include<thread>
using namespace std;
void proc(int &a)
{
    cout << "我是子线程,传入参数为" << a << endl;
    cout << "子线程中显示子线程id为" << this_thread::get_id()<< endl;
}
int main()
{
    cout << "我是主线程" << endl;
    int a = 9;
    thread th2(proc,a);//第一个参数为函数名，第二个参数为该函数的第一个参数，如果该函数接收多个参数就依次写在后面。此时线程开始执行。
    cout << "主线程中显示子线程id为" << th2.get_id() << endl;
    th2.join()；//此时主线程被阻塞直至子线程执行结束。
    return 0;
}
```

调用join()会清理线程相关的存储部分，这代表了join()只能调用一次

join()与detach()都是std::thread类的成员函数，是两种线程阻塞方法，两者的区别是是否等待子线程执行结束。

如果使用detach()，就必须保证线程结束之前可访问数据的有效性，使用指针和引用需要格外谨慎



### mutex互斥量

这样比喻：单位上有一台打印机（共享数据a），你要用打印机（线程1要操作数据a），同事老王也要用打印机(线程2也要操作数据a)，但是打印机同一时间只能给一个人用，此时，规定不管是谁，在用打印机之前都要向领导申请许可证（lock），用完后再向领导归还许可证(unlock)，许可证总共只有一个,没有许可证的人就等着在用打印机的同事用完后才能申请许可证(阻塞，线程1lock互斥量后其他线程就无法lock,只能等线程1unlock后，其他线程才能lock)。**那么，打印机就是共享数据，访问打印机的这段代码就是临界区，这个必须互斥使用的许可证就是互斥量**。



### **lock()与unlock()**

需要在进入临界区之前对互斥量lock，退出临界区时对互斥量unlock；当一个线程使用特定互斥量锁住共享数据时，其他的线程想要访问锁住的数据，都必须等到之前那个线程对数据进行解锁后，才能进行访问

程序实例化mutex对象m,本线程调用成员函数m.lock()会发生下面 2 种情况：

1. 如果该互斥量当前未上锁，则本线程将该互斥量锁住，直到调用unlock()之前，本线程一直拥有该锁。
2. 如果该互斥量当前被其他线程锁住，则本线程被阻塞,直至该互斥量被其他线程解锁，此时本线程将该互斥量锁住，直到调用unlock()之前，本线程一直拥有该锁

### **lock_guard**

原理是：声明一个局部的std::lock_guard对象，在其构造函数中进行加锁，在其析构函数中进行解锁。最终的结果就是：创建即加锁，作用域结束自动解锁。从而使用std::lock_guard()就可以替代lock()与unlock()。



### **unique_lock**

![img](assets/v2-b360bb0884ac5b575268c8d8d56a0818_720w.jpg)



### **condition_variable**

作用不是用来管理互斥量的，它的作用是用来同步线程，它的用法相当于编程中常见的flag标志（A、B两个人约定flag=true为行动号角，默认flag为false,A不断的检查flag的值,只要B将flag修改为true，A就开始行动）。

### wait

wait函数会自动调用 locker.unlock() 释放锁（因为需要释放锁，所以要传入mutex）并阻塞当前线程，本线程释放锁使得其他的线程得以继续竞争锁。一旦当前线程获得notify(通常是另外某个线程调用 notify_* 唤醒了当前线程)，wait() 函数此时再自动调用 locker.lock()上锁。

### notify_

cond.notify_one(): 随机唤醒一个等待的线程

cond.notify_all(): 唤醒所有等待的线程

## 异步线程

## 原子类型

## 生产者消费者模型

生产者用于生产数据，生产一个就往共享数据区存一个，如果共享数据区已满的话，生产者就暂停生产；消费者用于消费数据，一个一个的从共享数据区取，如果共享数据区为空的话，消费者就暂停取数据，且生产者与消费者不能直接交互



```c++
/*
生产者消费者问题
*/
#include <iostream>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include<Windows.h>
using namespace std;

deque<int> q;
mutex mu;
condition_variable cond;
int c = 0;//缓冲区的产品个数

void producer() { 
 int data1;
 while (1) {//通过外层循环，能保证生产不停止
  if(c < 3) {//限流
   {
    data1 = rand();
    unique_lock<mutex> locker(mu);//锁
    q.push_front(data1);
    cout << "存了" << data1 << endl;
    cond.notify_one();  // 通知取
    ++c;
   }
   Sleep(500);
  }
 }
}

void consumer() {
 int data2;//data用来覆盖存放取的数据
 while (1) {
  {
   unique_lock<mutex> locker(mu);
   while(q.empty())
    cond.wait(locker); //wait()阻塞前先会解锁,解锁后生产者才能获得锁来放产品到缓冲区；生产者notify后，将不再阻塞，且自动又获得了锁。
   data2 = q.back();//取的第一步
   q.pop_back();//取的第二步
   cout << "取了" << data2<<endl;
   --c;
  }
  Sleep(1500);
 }
}
int main() {
 thread t1(producer);
 thread t2(consumer);
 t1.join();
 t2.join();
 return 0;
}
```



## 线程池



## 惊群效应

