#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/4/23
-------------------------------------------------
"""

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from data.data_loader import HDF5DataLoader
import torchvision.transforms as transforms
from models import create_model
from util.visualizer import Visualizer
import os
import torch
import collections

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def main():
    opt = TrainOptions().parse()
    data_loader = HDF5DataLoader(opt)
    dataset = data_loader.load_data()

    # 得到训练样本的个数
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    total_steps = 0

    # epoch_count默认为1
    # niter默认为200，表示初始的学习率迭代的次数
    # niter_decay默认为200，表示这epoch 学习率都会线性衰减
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        print("----------------enumerate dataset begin----------------")
        for i, data in enumerate(dataset, 0):
            # pdb.set_trace()
            print("----------------enumerate dataset end----------------")
            print(data['A'].shape)


if __name__ == '__main__':
    main()