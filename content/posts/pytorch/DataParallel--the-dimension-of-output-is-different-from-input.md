---
title: "DataParallel  the Dimension of Output Is Different From Input"
date: 2021-12-15T16:24:02+08:00
draft: False
tags: [pytorch, CV]
categories: [pytorch]
comment:
    gitalk:
        id: "comments"
        enable: true
        owner: "birkhoffkiki"
        repo: "hugosite"
        clientId: "d6b047b05a271ae1181f"
        clientSecret: "8c774ac612503288bc0e50802e52a87e95d5a06c"
        admin: "birkhoffkiki"
---


# torch.nn.DataParallel

## 问题描述

当使用nn.DataParallel包裹模型后，模型输入的Batch维度和输出的batch维度不一致，此问题尤其会出现在多个输入的情况下，且其中某个输入为了节省资源被共用的情况下。 
举例说明情况：  
假设model正常接受两个输入$x_1 \in [N, C, H, W]$, $x_2\in [1, M, N]$, 且模型输出为$y_1 \in [N, C, H, W]$。 若model被nn.DataParallel包裹后且使用2个GPU时，假设$x_1, x_2$不变，此时输出将变为$y\in [N/2, C, H, W]$。  
```python
# 代码演示
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x1, x2):
        # implementation details
        return x1, x2


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    x1 = torch.randn((10, 3, 128, 128))
    x2 = torch.randn((1, 100, 200))
    model = Model()
    dp = nn.DataParallel(model)
    y1, y2 = dp(x1, x2)
    y3, y4 = model(x1, x2)
    print(torch.__version__)
    print('y1: {}, y2:{}'.format(y1.shape, y2.shape))
    print('y3: {}, y4:{}'.format(y3.shape, y4.shape))
```
output of code: 
> torch version: 1.5.0+cu101  
y1: torch.Size([5, 3, 128, 128]), y2:torch.Size([1, 100, 200])  
y3: torch.Size([10, 3, 128, 128]), y4:torch.Size([1, 100, 200]) 

## 原因分析  

主要原因在于输入中的某一个Tensor被复用，但其batch为1导致的。 当model被nn.DataParallel包裹后（假设其为dp)，在执行  

```python
y1, y2 = dp(x1, x2)
```

时，dp模型的$forward$函数会先被调用，我们来看一下其forward函数是怎么构成的:

```python
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))
            # 此处会对输入在dim=0上进行拆分，
            # 有几块卡就拆分为几份，因为我们的x2的第一个维度（dim=0）为一
            #无法再进行拆分，导致了此问题，可进一步探寻self.catter具体是如何工作的
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids) 
            # ... 其他code为节省空间，省略

```

上述代码中，$self.scatter$函数会对所有输入数据按比例进行拆分，有多少块卡每个输入就会在dim=0上被拆分为多少份。而例子中$x2$第一维为1，无法进一步拆分，所以虽然$x1$的拆分有两个，最终能够匹配的份数还是一。个人感觉与下列代码功能类似：  
```python
x1 = [1, 2]
x2 = [1]
for i, j in zip(x1, x2):
    # 只会循环一次
    print(i, j)
```

## 解决方案
将共用的Tensor的0维度翻倍，建议使用expand改变Tensor的view(view是Tensor的视图，可看成Tensor的维度)。expand不会对Tensor进行复制，相比于repeat能够节省内存。

```python

    x1 = torch.randn((10, 3, 128, 128))
    x2 = torch.randn((1, 100, 200))
    # 对x2的view进行调整，将其调整为GPU_num的倍数，以2为例
    x2 = x2.expand(2, -1, -1)
    model = Model()
    dp = nn.DataParallel(model)
    y1, y2 = dp(x1, x2)
    y3, y4 = model(x1, x2)
    print(torch.__version__)
    print('y1: {}, y2:{}'.format(y1.shape, y2.shape))
    print('y3: {}, y4:{}'.format(y3.shape, y4.shape))

```
outputs:
>torch version: 1.5.0+cu101  
y1: torch.Size([10, 3, 128, 128]), y2:torch.Size([2, 100, 200])  
y3: torch.Size([10, 3, 128, 128]), y4:torch.Size([2, 100, 200])  

## 额外探索实验

将x2的batch维度改为3

```python
    x1 = torch.randn((10, 3, 128, 128)).cuda()
    x2 = torch.randn((3, 100, 200)).cuda()
    model = Model()
    dp = nn.DataParallel(model)
    y1, y2 = dp(x1, x2)
    y3, y4 = model(x1, x2)
    print('torch version:', torch.__version__)
    print('y1: {}, y2:{}'.format(y1.shape, y2.shape))
    print('y3: {}, y4:{}'.format(y3.shape, y4.shape))
```

outputs:
> torch version: 1.5.0+cu101  
y1: torch.Size([10, 3, 128, 128]), y2:torch.Size([3, 100, 200])  
y3: torch.Size([10, 3, 128, 128]), y4:torch.Size([3, 100, 200])  

**注意** 此时$x1$,$x2$的划分为

```python
GPU0: x1 shape is [5, 3, 128, 128], x2 shape is [2, 100, 100]
GPU1: x2 shape is [5, 3, 128, 128], x2 shape is [1, 100, 100]
```
