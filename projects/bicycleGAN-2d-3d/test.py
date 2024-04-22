# coding=utf-8
import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
from util import util
import cv2
from options.parameters import EPOCH
import time
from PIL import Image
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# options
def main_bicycleGAN():
    opt = TestOptions().parse()
    # gpu_ids=None或者为''表示使用CPU
    # opt.gpu_ids= '0'
    opt.num_threads = 1   # test code only supports num_threads=1
    opt.batch_size = 1   # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle
    opt.no_encode=True
    opt.epoch=EPOCH

    # create dataset,加载数据
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    # 创建模型
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    print('Loading model %s' % opt.model)

    # 输出文件夹
    # 需要加载的模型的epoch
    img_root_dir = os.path.join(opt.results_dir, opt.phase+'_epoch_%d' % opt.epoch)
    # sample random z
    if opt.sync:
        z_samples = model.get_z_random(opt.n_samples, opt.nz)

    # test stage
    # opt.num_test=1

    acc_time=0.0
    for i, data in enumerate(islice(dataset, opt.num_test)):  #表示[0:opt.num_test]的数据
    # for i, data in enumerate(islice(dataset, opt.num_test,2*opt.num_test)):  #表示[0:opt.num_test]的数据
    # for i, data in enumerate(islice(dataset, 2*opt.num_test,4*opt.num_test)):  #表示[opt.num_test:2*opt.num_test]的数据
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))
        # 每次使用不同的噪声
        if not opt.sync:
            z_samples = model.get_z_random(opt.n_samples, opt.nz)

        # 得到input和target
        real_A, real_B=model.get_input_and_target()

        # 创建目录
        third_dir = os.path.join(img_root_dir, 'img_%03d' % i,'input')
        if not os.path.exists(third_dir):
            os.makedirs(third_dir)

        # 保存input
        input_img_path = os.path.join(third_dir, 'input.bmp')
        cv2.imwrite(input_img_path, util.tensor2im(real_A))

        third_dir = os.path.join(img_root_dir, 'img_%03d' % i, 'target/images')
        if not os.path.exists(third_dir):
            os.makedirs(third_dir)

        # 保存target，它的shape是4维的,如[1,128,128,128]
        img_num = real_B.shape[1]
        for j in range(img_num):
            target_img_path = os.path.join(third_dir, 'target_%03d.bmp' % j)
            im = util.tensor2im(real_B[0][j])
            cv2.imwrite(target_img_path, im)


        for nn in range(opt.n_samples):

            time_begin = time.time()

            fake_B= model.test(z_samples[[nn]])
            # fake_B = torch.cat((real_A,fake_B[::,1::,...]),dim=1)
            acc_time+=time.time()-time_begin

            # 为每个重建结果创建一个文件夹
            out_image_dir = os.path.join(img_root_dir, 'img_%03d'% i,'reconstructions/rec_%03d'% nn)
            if not os.path.exists(out_image_dir):
                os.makedirs(out_image_dir)

            img_num = fake_B.shape[1]
            for k in range(img_num):
                out_img_path = os.path.join(out_image_dir,'output_%03d.bmp' % k)
                im = util.tensor2im(fake_B[0][k],segflag=True)
                cv2.imwrite(out_img_path, im)

    # time_end = time.time()
    # print('averaged time per reconstruction is:%.4f s'% ((time_end-time_begin)/(opt.num_test*opt.n_samples)))
    print('averaged time per reconstruction is:%.4f s'% ((acc_time)/(opt.num_test*opt.n_samples)))

if __name__ == '__main__':
    main_bicycleGAN()
