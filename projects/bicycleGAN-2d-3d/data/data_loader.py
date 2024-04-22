# coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/4/22
-------------------------------------------------
"""

import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import h5py
import torch
from data.dataset import HDF5Dataset

import torchvision.transforms as transforms
class HDF5DataLoader():
    def __init__(self, opt):
        self.opt=opt

    def load_data(self):
        print('-------------loading data begin----------------')
        self.dataset = HDF5Dataset(self.opt,input_transform=transforms.Compose([
                          transforms.ToTensor()
                          ]))
        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.opt.batch_size,
                                          shuffle=not self.opt.serial_batches,  # 加载时，是否打乱图像顺序
                                          num_workers=int(self.opt.num_threads))
        print('-------------loading data end----------------')
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
