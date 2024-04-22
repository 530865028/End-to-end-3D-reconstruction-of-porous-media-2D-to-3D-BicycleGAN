import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import h5py

class AlignedDataset(BaseDataset):
    # 装饰器，申明为静态方法
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # opt.phase代表就是数据集文件夹的名字, choice=['train, val, test,], 默认值为'train'
        # dataroot为根目录
        # join之后，即为遍历dataroot/phase文件夹下面的图像，
        # 这和single_dataset.py里面有所区别，它是只有一个根目录，里面没有子文件夹。
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # 得到所有图片的路径，读图在__getitem__实现
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        with h5py.File(AB_path, "r") as f:
            img_2D = f['img_2D'][:]
            img_3D = f['img_3D'][:]
            # img_2D = f['img_2D'][()]
            # img_3D = f['img_3D'][()]

        # A,B需要为3维
        A = torch.Tensor(img_2D)
        # [0,255] -->[-1,1]
        A = A.div(255.0).sub(0.5).div(0.5)
        # 增加一个维度
        A = A.unsqueeze(0)
        B = torch.Tensor(img_3D)
        B = B.div(255.0).sub(0.5).div(0.5)

        # print("shape of A:",A.shape)
        # print("shape of B:",B.shape)

        # 这里主要是保留了BiCycleGAN的写法

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc


        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}


        # A,B需要为3
        # AB = Image.open(AB_path).convert('RGB')
        # 读图，单通道

        # 增加一维为通道数
        # A.unsqueeze_(0)
        # B.unsqueeze_(0)

        # print('\n--------shape of A-----------\n', A.shape)
        # print(A.min().data.numpy())
        # print(A.max().data.numpy())
        # print(A)

        # print(B.min().data.numpy())
        # print(B.max().data.numpy())
        # print('\n--------shape of B-----------\n', A.shape)

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
