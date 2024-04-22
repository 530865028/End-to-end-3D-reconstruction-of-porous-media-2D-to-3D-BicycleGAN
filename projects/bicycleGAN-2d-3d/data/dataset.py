import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import h5py
import torch


# 判断是否为 .hdf5 或者.h5文件
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])


##从h5文件中加载图像数据，并做归一化
def load_img(filepath):
    print('--------load image--------')
    with h5py.File(filepath, "r") as f:
        # img_2D = f['img_2D'][:]
        # img_3D = f['img_3D'][:]
        img_2D = f['img_2D'][()]
        img_3D = f['img_3D'][()]
        print('shape img_2D', img_2D.shape)
        print('shape img_3D', img_3D.shape)

    # A,B需要为3维
    # A = torch.from_numpy(img_2D)
    A=torch.Tensor(img_2D)
    print("shape of A:", A.shape)
    # [0,255] -->[-1,1]
    # A=A.float()
    A = A.div(255).sub(0.5).div(0.5)
    # 增加一个维度
    A = A.unsqueeze(0)

    B = torch.from_numpy(img_3D)
    B=B.float()
    print("shape of B:", B.shape)

    # 这里主要是保留了BiCycleGAN的写法
    return {'A': A, 'B': B,
            'A_paths': filepath, 'B_paths': filepath}

# 对HDF5Dataset的进行若干操作(这里在load_img里面做的)：遍历所有的h5文件保存到self.image_filenames中；加载其中一项；得到数据集的个数
class HDF5Dataset(data.Dataset):
    def __init__(self, opt,input_transform=None, target_transform=None):
        super(HDF5Dataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        # opt.phase代表就是数据集文件夹的名字, choice=['train, val, test,], 默认值为'train'
        # dataroot为根目录
        # join之后，即为遍历dataroot/phase文件夹下面的图像，
        # 这和single_dataset.py里面有所区别，它是只有一个根目录，里面没有子文件夹。
        self.dir_AB = join(opt.dataroot, opt.phase)
        # 得到按升序排序的所有图片的路径，读图在__getitem__实现
        # 这句代码的写得我有点懵逼。

        self.AB_paths = sorted([join(self.dir_AB, x) for x in listdir(self.dir_AB) if is_image_file(x)])
        self.input_transform = input_transform
        self.target_transform = target_transform

    ##从h5文件中加载图像数据，并做归一化
    def load_img(self, filepath):
        with h5py.File(filepath, "r") as f:
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

        print("shape of A:",A.shape)
        print("shape of B:",B.shape)

        # 这里主要是保留了BiCycleGAN的写法
        return {'A': A, 'B': B,
                'A_paths': filepath, 'B_paths': filepath}

    def __getitem__(self, index):
        sample = self.load_img(self.AB_paths[index])
        return sample

    def __len__(self):
        return len(self.AB_paths)
