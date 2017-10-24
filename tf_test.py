# -*- coding:utf-8 -*-
#首先引入所需要的基本包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
 
#读matlab文件
from scipy.io import loadmat as load 
 
 
#读取数据
train= load('/Users/Loj/Desktop/train_32x32.mat')
test= load('/Users/Loj/Desktop/test_32x32.mat')
 
#这时后，我们可以打印数据的shape进行观察。
print(train['X'].shape)
print(train['y'].shape)
 
print(test['X'].shape)
print(test['y'].shape)
 
"""
结果为：
(32,32, 3, 73257)
(73257,1)
(32,32, 3, 26032)
(26032,1)
训练集为73257个样本，测试集26032个样本，每个样本是32*32像素，3个像素通道，后面我们将其转化为1个通道，用灰度图显示，另外会将X样本量改为第一位置，符合我们平时处理的规格。

"""

"""
下面我们定义三个函数reformat(samples,labels) normalize(samples)，inspect(dataset, labels, i)：
 
reformat用来改变原始数据格式，并对lable进行独热编码（one-hot encoding）。
 
normalize用来对数据进行灰度化（
将三色通道转化为单色通道），把数据映射到 -1.0 ~ +1.0之间。
 
inspect将图片显示有助于观察。
"""
def reformat(samples, labels):
    # 改变原始数据的形状
    # （ 0       1       2     3）                   （ 3       0      1      2）
    # (图片高，图片宽，通道数，图片数) -> (图片数，图片高，图片宽，通道数)
    new = np.transpose(samples, (3, 0,1, 2)).astype(np.float32)

    # labels 变成 one-hot encoding,[2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # digit 0 ,  represented as 10
    # labels 变成 one-hot encoding,[10] -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = np.array([x[0] for x in labels])          
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] =1.0
        else:
            one_hot[num]= 1.0
        one_hot_labels.append(one_hot)
    labels =np.array(one_hot_labels).astype(np.float32)
    return new, labels
 
def normalize(samples):
    """
    灰度化: 从三色通道 -> 单色通道     省内存 ，加快训练速度
    (R + G + B) / 3
    将图片从 0 ~ 255 线性映射到-1.0 ~ +1.0
    """
    a = np.add.reduce(samples,keepdims=True, axis=3)  # shape (图片数，图片高，图片宽，通道数)，将samples沿着原来格式相加
    a = a/3.0
    return a/128.0 - 1.0
 
 
def inspect(dataset, labels, i):
    # 将图片显示出来
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset =dataset.reshape(shape[0], shape[1], shape[2])
    print(labels[i])
    plt.imshow(dataset[i])
    plt.show()
 
 
#这时候我们就可以测试一下，打印一张图片看看
train_samples = train['X']
train_labels = train['y'] 
_train_samples,_train_labels = reformat(train_samples, train_labels)
 
inspect(_train_samples,_train_labels, 123)
 

#结果为:
# [ 0. 0.  0.  0. 0.  0.  1. 0.  0.  0.]

