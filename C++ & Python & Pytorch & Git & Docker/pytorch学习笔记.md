# Pytorch 学习笔记

---

> 资料：
>
> [PyTorch深度学习：60分钟入门(Translation)](<https://zhuanlan.zhihu.com/p/25572330>)



## 1. 数据处理与读取

[官方教程](<https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>)

自定义Transforms类，transforms.Compose对多个操作进行组合

```python
class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}
class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
    
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
```

使用torchvision中的transform

```python
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
```



## 2.  动态变化网络的权值共享

支持动态图，每次调用的DynamicNet都在变化

```python
# -*- coding: utf-8 -*-
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 3. 模型的保存与加载

+ torch.save：将序列化对象保存到磁盘。这个函数使用Python的pickle实用程序进行序列化。使用该函数可以保存各种对象的模型、张量和字典

+ torch.load:  使用pickle的反pickle工具将pickle的对象文件反序列化到内存中。该函数还方便设备将数据加载到其中

+ torch.nn.Module.load_state_dict: 使用反序列化的state_dict加载模型的参数字典

  

  state_dict只是一个Python dictionary对象，它将每个层映射到它的参数张量;

  注意，只有具有可学习参数的层(卷积层、线性层等)和注册缓冲区(batchnorm的running_mean)在模型的state_dict中有条目

  优化器torch.optim中也有state_dict，包含了优化器状态参数

输出示例：

```python
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

输出：

```python
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```

保存

```python
torch.save(model.state_dict(), PATH)
```

加载：

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
# 请记住，在运行推理之前，必须调用model.eval()将dropout和批处理规范化层设置为评估模式
# 如果不这样做，将产生不一致的推理结果
model.eval()
```

加载或保存 Checkpoint for Inference 并且/或者继续训练

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```

```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or - 继续训练
model.train()
```

## 4. tensorboardX可视化

```python
# 使用tensorboard可视化
if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")
```



## 5. model.eval

model.train与model.eval的区别：

主要是针对model 在训练时和评价时不同的 **Batch Normalization**  和  **Dropout 方法**模式。

eval（）时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值

## 6. 什么是param_groups?

optimizer通过param_group来管理参数组.param_group中保存了参数组及其对应的学习率,动量等等.所以我们可以通过更改param_group[‘lr’]的值来更改对应参数组的学习率

```python
# 有两个`param_group`即,len(optim.param_groups)==2
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
 
#一个参数组
optim.SGD(model.parameters(), lr=1e-2, momentum=.9)
```

## 7. DataLoader

[pytorch 函数DataLoader](<https://www.e-learn.cn/content/qita/850153>)

Dataset类一次调用getitem只返回一个样本

DataLoader类用于minibatch的训练，是一个可迭代的对象

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, 
           num_workers=0, collate_fn=default_collate, pin_memory=False,
           drop_last=False)
# 说明
'''
dataset：加载的数据集(Dataset对象) 
batch_size：batch size 
shuffle:：是否将数据打乱 
sampler： 样本抽样，后续会详细介绍 
num_workers：使用多进程加载的进程数，0代表不使用多进程 
collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可 
pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些 
drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
'''
```

## 8. Pytorch中的下划线

in-place类型是指，但在一个tensor上操作了之后，是直接修改了这个tensor，还是返回一个新的tensor，而旧的tensor并不修改

pytorch中，一般来说，如果对tensor的一个函数后加上了下划线，则表明这是一个in-place类型，会直接修改原始值

## 9. item()， data

pytorch使用.item()获取tensor的值

如果x是一个变量，则x.data是Tensor

## 10. contiguous()与view()

contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。 
一种可能的解释是： 
有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。 

判断是否contiguous用torch.Tensor.is_contiguous()函数。

## 11. detach

