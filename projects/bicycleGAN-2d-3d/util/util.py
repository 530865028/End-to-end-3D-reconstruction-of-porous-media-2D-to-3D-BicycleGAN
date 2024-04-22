from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle
import torchvision.transforms as transforms
from skimage import transform
import skimage.io as io

from options.parameters  import GRAY_THRESHOLD
# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array


def PILimg2tensor(img):
    img_tensor=torch.from_numpy(img)
    img_tensor=img_tensor.float()
    # print(img_tensor)
    img_tensor=img_tensor.div(255.0).sub(0.5).div(0.5)
    img_tensor=img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    # print(img_tensor)
    return img_tensor

# Tensor的格式为[1,C,H,W]
# 转换后的格式为[H,W,C]
# 单通道为[H,W]
def tensor2im(input_image, imtype=np.uint8, segflag=True):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = None
    if input_image.dim() == 2:
        image_numpy = image_tensor.cpu().float().numpy()  # [H,W]
    elif input_image.dim() == 3:
        image_numpy = image_tensor[0].cpu().float().numpy()  # [H,W]

    elif input_image.dim() == 4:
        image_numpy = image_tensor[0].cpu().float().numpy()  # [1,H,W]
        # [1,H,W] --> [H,W]
        image_numpy = np.squeeze(image_numpy, axis=0)  # 去掉第一个维度（单通道）

    image_numpy = (image_numpy + 1) / 2.0 * 255.0  # 转为0-255
    # 是否分割
    if segflag:
        image_numpy = image_numpy > GRAY_THRESHOLD  # out_img的值为True或者False
        image_numpy = (image_numpy + 0).astype(np.uint8) * 255  # out_img+0将True或者False转为1或者0
    # return np.uint8(image_numpy)  等效
    return image_numpy.astype(imtype)


# added by fjx 20190423
def tensor2im_Segment(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = None
    if input_image.dim() == 2:
        image_numpy = image_tensor.cpu().float().numpy()  # [H,W]
    elif input_image.dim() == 3:
        image_numpy = image_tensor[0].cpu().float().numpy()  # [H,W]

    elif input_image.dim() == 4:
        image_numpy = image_tensor[0].cpu().float().numpy()  # [1,H,W]
        # [1,H,W] --> [H,W]
        image_numpy = np.squeeze(image_numpy, axis=0)  # 去掉第一个维度（单通道）

    image_numpy = (image_numpy + 1) / 2.0 * 255.0  # 转为0-255
    image_numpy = image_numpy > GRAY_THRESHOLD  # out_img的值为True或者False
    image_numpy = (image_numpy + 0).astype(np.uint8) * 255  # out_img+0将True或者False转为1或者0

    return image_numpy.astype(imtype)

def tensor2vec(vector_tensor):
    numpy_vec = vector_tensor.data.cpu().numpy()
    if numpy_vec.ndim == 4:
        return numpy_vec[:, :, 0, 0]
    else:
        return numpy_vec


def pickle_load(file_name):
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def interp_z(z0, z1, num_frames, interp_mode='linear'):
    zs = []
    if interp_mode == 'linear':
        for n in range(num_frames):
            ratio = n / float(num_frames - 1)
            z_t = (1 - ratio) * z0 + ratio * z1
            zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    if interp_mode == 'slerp':
        z0_n = z0 / (np.linalg.norm(z0) + 1e-10)
        z1_n = z1 / (np.linalg.norm(z1) + 1e-10)
        omega = np.arccos(np.dot(z0_n, z1_n))
        sin_omega = np.sin(omega)
        if sin_omega < 1e-10 and sin_omega > -1e-10:
            zs = interp_z(z0, z1, num_frames, interp_mode='linear')
        else:
            for n in range(num_frames):
                ratio = n / float(num_frames - 1)
                z_t = np.sin((1 - ratio) * omega) / sin_omega * z0 + np.sin(ratio * omega) / sin_omega * z1
                zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    return zs


def save_image(image_numpy, image_path):

    image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path, 'JPEG', quality=100)   # 'JPEG'模式下，图像会被压缩
    image_pil.save(image_path,'BMP')  # 这里改为'BMP'模式，即使保存为.jpeg，也不会被压缩


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)
                             ).repeat(1, in_feat.size()[1], 1, 1)
    return in_feat / (norm_factor + eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    return torch.mean(torch.sum(in0_norm * in1_norm, dim=1))


def save_image_3D(path, image3D):
    image3D = np.array(image3D)
    if image3D.ndim != 3:
        raise Exception("the dimension of image3D should be 3.")
    if not os.path.exists(path):
        os.makedirs(path)
    for i, img in enumerate(image3D):
        out_img_path = os.path.join(path, 'output_%03d.bmp' % i)
        io.imsave(out_img_path, img)

# order的值为0-5，0表示最近邻插值(二值图像)，anti_aliasing表示下采样的时候不用高斯模糊
# target_size应该是一个tuple
def resize_image_3D(image3D, target_size, order=0):
    image3D=np.array(image3D)
    # preserve_range=True即不做归一化，anti_aliasing=False表示下采样之前不加高斯滤波
    resized_img=transform.resize(image3D, target_size, order,preserve_range=True,anti_aliasing=False)
    # resized_img的值为0-1
    # scaled_img=(255*resized_img).astype(np.uint8)
    scaled_img=resized_img.astype(np.uint8)
    return scaled_img

# scale应该是一个数,大于1表示放大，小于1表示缩小
# anti_aliasing表示下采样的时候不用高斯模糊
def rescale_image_3D(image3D, scale, order=0):
    image3D=np.array(image3D)
    # preserve_range=True即不做归一化，anti_aliasing=False表示下采样之前不加高斯滤波，multichannel=False表示不是多通道的彩色图像
    scaled_img=transform.rescale(image3D, scale, order,preserve_range=True,multichannel=False,anti_aliasing=False)
    # scaled_img的值为0-1
    scaled_img=scaled_img.astype(np.uint8)
    return scaled_img

class MyRandomTranform(object):
    def __init__(self):
        super(MyRandomTranform, self).__init__()

    def RandomTransform(self, input):

        ##rotate和transpose的变换没有添加
        num=np.random.randint(0,7)
        # print("num=",num)
        trans_result=None
        if (num == 0):
            trans_result = input
        elif(num==1):
            trans_result = np.flip(input, 0)
        elif(num==2):
            trans_result = np.flip(input, 1)
        elif (num == 3):
            trans_result = np.flip(input, 2)
        elif (num == 4):
            trans_result= np.flip(input, (0, 1))
        elif (num == 5):
            trans_result = np.flip(input, (0, 2))
        elif (num == 6):
            trans_result = np.flip(input, (1, 2))
        else:
            raise Exception("no suitable transformation!")

        return trans_result

    def __call__(self, input):
        return self.RandomTransform (input)