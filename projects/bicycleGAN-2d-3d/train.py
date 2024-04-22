import time

from options.train_options import TrainOptions
from data import CreateDataLoader
from data.data_loader import HDF5DataLoader

from models import create_model
from util.visualizer import Visualizer
import os
import torch
import collections
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(123)

# torch.autograd.set_detect_anomaly(True)
#TODO 添加保存中间3D结构的功能
def main():

    # 指定训练时的参数
    opt = TrainOptions().parse()
    # 创建 dataloader

    data_loader = CreateDataLoader(opt)
    # # 加载数据
    dataset = data_loader.load_data()

    # data_loader=HDF5DataLoader(opt)
    # dataset=data_loader.load_data()

    # 得到训练样本的个数
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    # 创建模型(bicycle_gan_model'或者'pix2pix_model)
    model = create_model(opt)
    # 设置是开始训练还是加载网络来进行测试
    model.setup(opt)
    # visdom来进行可视化

    print(model.netG)
    # print(model.netD)
    # print(model.netE)
    # a=torch.rand(2,128,128,128).cuda()
    # print(model.netD(a)[0].shape)
    # return

    visualizer = Visualizer(opt)
    # 记录总的steps,一个batch就记为一个step
    total_steps = 0

    # epoch_count默认为1
    # niter默认为200，表示初始的学习率迭代的次数
    # niter_decay默认为200，表示这epoch 学习率都会线性衰减
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            # print('-------processing data %d-----------'%i)
            # 每一个batch的起始时间
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            # data是字典，分别feed给input和target
            model.set_input(data)
            # 判断是否训练（包括特殊情况：real_A.size(0) != self.opt.batch_size，则continue）
            if not model.is_train():
                continue
            # 更新参数
            model.optimize_parameters()

            # 显示结果
            # opt.display_freq=400,opt.print_freq=100
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                # 将需要观察的图像在这个时刻的值显示出来
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            # 显示Loss
            if total_steps % opt.print_freq == 0:
                # 将需要观察的loss在这个时刻的值显示出来
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            # 保存
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        # 保存
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # 更新学习率，随着epoch的变化，学习率在递减
        # 递减规律（线性的）  lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
        model.update_learning_rate()

if __name__ == '__main__':

    main()
