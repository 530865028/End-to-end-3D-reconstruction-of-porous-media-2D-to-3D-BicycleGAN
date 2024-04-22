#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/2/24
-------------------------------------------------
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import h5py

def conv_init(conv):
    weights = torch.Tensor([[256, 128, 64], [32, 16, 8], [4, 2, 1]])
    weights = weights.expand(1, 1, 3, 3)
    # print(weights)
    conv.weight.data = weights

np_img=np.array([[0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 1, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1],
              [1, 1, 0, 1, 1, 1, 1],
              [0, 0, 1, 1, 1, 1, 1],
              [1, 1, 0, 1, 1, 1, 1],
              [0, 1, 0, 1, 1, 1, 1]],dtype=np.float32)
print(np_img)
print(np_img.shape)

img=torch.from_numpy(np_img)
print(img)
img=img.unsqueeze_(0).unsqueeze_(0)
MyTemplate_Conv=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=0,dilation=1,bias=False)

MyTemplate_Conv.apply(conv_init)

Hist=MyTemplate_Conv(img)  #计算统计直方图[Batch,C,H-2,W-2]
Hist=Hist.reshape(-1)
Hist=Hist.int()
Count=torch.bincount(Hist.cpu(),minlength=512)#计数

#-------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,没有小数------------#
Count = Count.float()
# -------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,没有小数------------#
Count=Count/len(Hist)
np_data=Count.data.numpy()
print(np_data)
np.savetxt('Count_Hist.txt',np_data,fmt='%0.4f')
print('数据保存当前文件夹下的Count_Hist.txt中！')