---
title: "Bilinear_interpolation 原理及实现"
date: 2021-12-09T16:28:12+08:00
draft: false
tags: ["image processing", "CV"]
categories: [image processing]
---
# 图像插值
:smile: 本文参考[此博客](https://theailearner.com/2018/12/29/image-processing-bilinear-interpolation/) 


# pytorch interpolation
torch.nn.functional.interpolate 可能将正像素插值为负像素（待查明）