# coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2018/7/26
-------------------------------------------------
"""

import cv2
import os
import glob
import numpy as np
import h5py
import os
import time
import random
import argparse
import shutil
import sys
from skimage import io, transform
from tqdm import tqdm

# 添加上一级目录的路径，然后import my_util
sys.path.append('..')
import my_util
from util import util

parser = argparse.ArgumentParser(description='create traning set or testing set')

parser.add_argument('--input_image_path', required=False, default='', help='input image path')
parser.add_argument('--output_path', required=False, default='train', help='output name of training data')
parser.add_argument('--CLASS', required=False, default='manmade_sandstone',
                    choices=['sandstone', 'manmade_sandstone', 'battery'], help='class of dataset')

opt = parser.parse_args()

"""
#函数:ImagesTohdf5(image_path, remain_area_x, remain_area_y, output_path, output_name)
#功能：将图像数据转为H5文件，用作训练集
#参数：
#image_path：输入图像文件夹路径，如"F:/1"
#remain_area_x：等于图像宽度-X方向保留的区域，值为a则X方向保留W-a
#remain_area_y：Y方向保留的区域，值为b则Y方向保留b
#output_path：h5文件的输出路径
#output_name：h5文件的文件名

"""

"""
#函数:Image3D2hdf5(image_3D,remain_area_x,remain_area_y,output_path,output_name)
#功能：将图像数据转为H5文件，用作训练集
#参数：
#image_3D：输入图像数据，numpy格式
#remain_area_x：等于图像宽度-X方向保留的区域，值为a则X方向保留W-a
#remain_area_y：Y方向保留的区域，值为b则Y方向保留b
#output_path：h5文件的输出路径
#output_name：h5文件的文件名
"""


# 图像归一化，由[0,255]->[-1,1]
# 返回归一化后的值
def ImageNormalization(Image):
    temp = Image / 255
    temp = temp / 255
    temp = (temp - 0.5) / 0.5
    return temp  # 归一化到[-1,1]


# create traning set
def create_training_set():
    output_name = 'train'
    CreateDataset(image_path=opt.input_image_path,
                  remain_area_x=opt.remain_area_x,
                  remain_area_y=opt.remain_area_y,
                  output_path=opt.ouput_base_path,
                  output_name=output_name)

    print("training set has been succesfully created")


# create testing set

def create_testing_set():
    output_name = 'test'
    CreateDataset(image_path=opt.input_image_path,
                  remain_area_x=opt.remain_area_x,
                  remain_area_y=opt.remain_area_y,
                  output_path=opt.ouput_base_path,
                  output_name=output_name)

    print("testing set has been succesfully created")


# output_name=choice['train','val','test']

def CreateDataset(image_path, remain_area_x=128, remain_area_y=20, output_path='', output_name='train'):
    if image_path is None or not os.path.exists(image_path):
        raise Exception("path '%s' does not exist" % image_path)

    output_path = os.path.join(output_path + "_%d" % (remain_area_y), output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('====output path of dataset is :', output_path)

    all_image_path = glob.glob(os.path.join(image_path, "*.bmp"))  # 得到当前文件夹下的所有bmp格式的图片
    if (len(all_image_path) == 0):
        raise Exception("this directory has no images")

    for i in range(len(all_image_path)):
        lable = cv2.imread(all_image_path[i], flags=0)  # 这里不做归一化，在数据导入的时候做
        # x= lable.copy()  # 这里用copy()函数将x_data_temp与y_data_temp分离，互不影响。
        # x[remain_area_y:, remain_area_x:] = 127  # 将剩下的区域置为127

        input = np.zeros_like(lable) + 127  # 全部赋值为127

        input[0:remain_area_y, 0: remain_area_x] = lable[0:remain_area_y, 0: remain_area_x]

        combine_image = np.concatenate((input, lable), axis=1)
        Image_path = os.path.join(output_path, 'Image%04d.bmp' % i)

        cv2.imwrite(Image_path, combine_image)


##########################专门为了生成4个小区域数据而写#########################################
def Create_Training_Set_For_4_subareas():
    output_name = 'train'
    CreateDataset_For_4_subareas(image_path=opt.input_image_path,
                                 output_path=opt.ouput_base_path,
                                 output_name=output_name)

    print("training set has been succesfully created")


def Create_Testing_Set_For_4_subareas():
    output_name = 'test'
    CreateDataset_For_4_subareas(image_path=opt.input_image_path,
                                 output_path=opt.ouput_base_path,
                                 output_name=output_name)

    print("testing set has been succesfully created")


def CreateDataset_For_4_subareas(image_path, output_path='', output_name='train'):
    if image_path is None or not os.path.exists(image_path):
        raise Exception("path '%s' does not exist" % image_path)

    output_path = os.path.join(output_path, output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('output path of dataset is :', output_path)

    all_image_path = glob.glob(os.path.join(image_path, "*.bmp"))  # 得到当前文件夹下的所有bmp格式的图片
    if (len(all_image_path) == 0):
        raise Exception("this directory has no images")

    SubAreaHeight = 13
    SubAreaWidth = 13
    coordinates = [(35, 35), (35, 80), (80, 35), (80, 80)]

    for i in range(len(all_image_path)):
        lable = cv2.imread(all_image_path[i], flags=0)  # 这里不做归一化，在数据导入的时候做
        H, W = lable.shape
        # x= lable.copy()  # 这里用copy()函数将x_data_temp与y_data_temp分离，互不影响。
        input = np.zeros_like(lable) + 127  # 全部赋值为127

        for (x, y) in coordinates:
            # 越界判断
            if x >= H or y >= W or x + SubAreaHeight >= H or y + SubAreaWidth >= W:
                raise Exception("图像访问越界！")

            input[x:x + SubAreaHeight, y:y + SubAreaWidth] = lable[x:x + SubAreaHeight, y:y + SubAreaWidth]

        combine_image = np.concatenate((input, lable), axis=1)
        Image_path = os.path.join(output_path, 'Image%04d.bmp' % i)
        cv2.imwrite(Image_path, combine_image)


##########################专门为了生成4个小区域数据而写#########################################

##########################组合多个数据集#########################################

# 拷贝文件,2018-11-02
def MyCopyFile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


def Combine_Samples(input_path, phase, combined_indexes, NUM=200, dir_name=None):


    out_img_nums=NUM*len(combined_indexes)
    all_output_image_path=[os.path.join(input_path, phase + '_256_combined', dir_name,'sample_%04d.h5' % j) for j in range(out_img_nums)]
    random.seed(123)  # 设置种子点，保证每次的shuffle是一致的
    random.shuffle(all_output_image_path)  # in-place替换output_image_path

    Length_of_Training = 0
    for i in combined_indexes:
        image_path = os.path.join(input_path, phase + "_%d_256_transform" % (i))
        print("------processing %d data set------" % (i))
        if not os.path.exists(image_path):
            raise RuntimeError(('path %s do not exit') % (image_path))

        all_image_path = glob.glob(os.path.join(image_path, "*.h5"))  # 得到当前文件夹下的所有bmp格式的图片

        random.seed(123)  # 设置种子点，保证每次的shuffle是一致的
        random.shuffle(all_image_path)  # in-place替换all_image_path
        len_img=len(all_image_path)
        if len_img< NUM:
            NUM=len_img
        training_image_path = all_image_path[:NUM]  # 取前半部分作为训练集

        for j in range(len(training_image_path)):
            # output_image_path = os.path.join(input_path, phase + '_combined', dir_name,
            #                                           'sample_%04d.h5' % (Length_of_Training + j))
            output_image_path=all_output_image_path[Length_of_Training + j]
            MyCopyFile(training_image_path[j], output_image_path)

        Length_of_Training += len(training_image_path)  # 为了后续更新标号


def Combine_Samples_For_Training(input_path, phase, combined_indexes, NUM=200, dir_name="train"):
    Combine_Samples(input_path, phase, combined_indexes, NUM, dir_name)


def Combine_Samples_For_Testing(input_path, phase, combined_indexes, NUM=100, dir_name="test"):
    Combine_Samples(input_path, phase, combined_indexes, NUM, dir_name)

##########################组合多个数据集#########################################


##########################专门为了生成随机几个修复区域的数据集而写#########################################

def Rand_Generate_Several_Mask_Area_For_Training():
    output_name = 'train'
    CreateDataset_For_Several_Mask_Area(image_path=opt.input_image_path,
                                        output_path=opt.ouput_base_path,
                                        output_name=output_name)

    print("train set has been succesfully created")


def Rand_Generate_Several_Mask_Area_For_Testing():
    output_name = 'test'
    CreateDataset_For_Several_Mask_Area(image_path=opt.input_image_path,
                                        output_path=opt.ouput_base_path,
                                        output_name=output_name)

    print("test set has been succesfully created")


def CreateDataset_For_Several_Mask_Area(image_path, output_path='', output_name='train'):
    if image_path is None or not os.path.exists(image_path):
        raise Exception("path '%s' does not exist" % image_path)

    output_path = os.path.join(output_path, output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('output path of dataset is :', output_path)

    all_image_path = glob.glob(os.path.join(image_path, "*.bmp"))  # 得到当前文件夹下的所有bmp格式的图片
    if (len(all_image_path) == 0):
        raise Exception("this directory has no images")

    SubAreaHeight = 30
    SubAreaWidth = 30
    coordinates = [(20, 20), (20, 75), (75, 20), (75, 75)]

    np.random.seed(123)
    random.seed(123)

    for i in range(len(all_image_path)):
        lable = cv2.imread(all_image_path[i], flags=0)  # 这里不做归一化，在数据导入的时候做
        H, W = lable.shape
        input = lable.copy()  # 这里用copy()函数将x_data_temp与y_data_temp分离，互不影响。
        # input=np.zeros_like(lable)+127  # 全部赋值为127
        # Num = np.random.randint(1, 4)
        Num = random.randint(1, 4)
        # print('选择的区域个数为：',Num)
        rand_coordinates = [random.choice(coordinates) for _ in range(Num)]

        for (x, y) in rand_coordinates:
            # 越界判断
            if x >= H or y >= W or x + SubAreaHeight >= H or y + SubAreaWidth >= W:
                raise Exception("图像访问越界！")

            input[x:x + SubAreaHeight, y:y + SubAreaWidth] = 127

        combine_image = np.concatenate((input, lable), axis=1)
        Image_path = os.path.join(output_path, 'Image%04d.bmp' % i)
        cv2.imwrite(Image_path, combine_image)


##########################专门为了生成随机几个修复区域的数据集而写#########################################


def main_for_2D_to_2D():
    # 通常的参数
    REMAIN_Y = 20  # 保留区域的高度
    REMAIN_X = 128  ##保留区域的长度

    ##################通用数据集制作方式############################
    opt.CLASS = 'anisotropic'  # choices=['silicon','sandstone','manmade_sandstone','battery','sandstone_low_posority']
    dataset_path = '../datasets/original_imagesets/'
    # opt.output_path = os.path.join('../datasets', opt.CLASS+'_square')
    opt.output_path = os.path.join('../datasets', opt.CLASS)

    print(opt)
    print("\n")

    # for index in [8]:
    for index in [10, 20]:
        REMAIN_Y = index
        opt.remain_area_y = REMAIN_Y
        opt.remain_area_x = REMAIN_X

        opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'training_images')

        create_training_set()  # 生成训练集
        opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'testing_images')
        create_testing_set()  # 生成测试集

    ##################通用数据集制作方式############################

    ##################专门为左上角小方形的数据集制作方式############################

    # 保留一个方形
    # REMAIN_Y = 26  # 保留区域的高度
    # REMAIN_X = 26  ##保留区域的长度
    #
    # opt.CLASS='silicon'   # choices=['silicon','sandstone','manmade_sandstone','battery','sandstone_low_posority']
    # dataset_path = '../datasets/original_imagesets/'
    # opt.output_path = os.path.join('../datasets', opt.CLASS+'_square')
    #
    # # print(opt)
    # # print("\n")
    #
    # # for index in [8]:
    # # for index in [1,3,5,10,20]:
    #
    # opt.remain_area_y=REMAIN_Y
    # opt.remain_area_x=REMAIN_X
    #
    # opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'training_images')
    # create_training_set()  #生成训练集
    #
    # opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'testing_images')
    # create_testing_set() #生成测试集

    ##################专门为左上角小方形的数据集制作方式############################

    #################组合数据集###############################
    # Combine_Images_For_Training()
    # Combine_Images_For_Testing()

    #################组合数据集###############################

    ##########################专门为了生成4个小区域数据而写#######################

    # opt.CLASS = 'silicon'  # choices=['silicon','sandstone','manmade_sandstone','battery','sandstone_low_posority']
    # dataset_path = '../datasets/original_imagesets/'
    # opt.output_path = os.path.join('../datasets', opt.CLASS+'_4_subareas')
    # print(opt)
    # print("\n")
    # opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'training_images')
    # Create_Training_Set_For_4_subareas()  #生成训练集
    # opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'testing_images')
    # Create_Testing_Set_For_4_subareas() #生成测试集

    ##########################专门为了生成4个小区域数据而写#######################

    # opt.CLASS = 'anisotropic'  # choices=['silicon','sandstone','manmade_sandstone','battery','sandstone_low_posority']
    # dataset_path = '../datasets/original_imagesets/'
    # opt.output_path = os.path.join('../datasets', opt.CLASS + '_inpainting')
    # print(opt)
    # print("\n")
    # opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'training_images')
    # Rand_Generate_Several_Mask_Area_For_Training()  # 生成训练集
    # opt.input_image_path = os.path.join(dataset_path, opt.CLASS, 'testing_images')
    # Rand_Generate_Several_Mask_Area_For_Testing()  # 生成测试集


def create_data_for_2D_to_3D(input_dir, out_dir, cut_size=128, scale=None,target_size =None,transform=True):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("path '%s' does not exist" % input_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('output path of dataset is :', out_dir)
    all_image_path = sorted(glob.glob(os.path.join(input_dir, "*.bmp")))  # 得到当前文件夹下的所有bmp格式的图片
    if (len(all_image_path) == 0):
        raise Exception("this directory has no images")
    print("图像数量为：", len(all_image_path))
    # 保存3D图像
    img_3D_data = []
    for i in range(len(all_image_path)):
        img = cv2.imread(all_image_path[i], flags=0)  # 这里不做归一化，在数据导入的时候做
        #### img = io.imread(all_image_path[i], as_gray=True)  #读取出来是double类型[0,1]
        img_3D_data.append(img)
        # H, W = img.shape
        # input = img.copy()  # 这里用copy()函数将x_data_temp与y_data_temp分离，互不影响。

    # 列表转成numpy
    img_3D_data = np.array(img_3D_data)
    L, H, W = img_3D_data.shape
    print("shape of img_3D_data", img_3D_data.shape)
    # stride = 48
    stride = 40
    # stride = 1
    sample_l, sample_h, sample_w = (cut_size, cut_size, cut_size)
    sample_num = 0

    range_L = range(0, L - sample_l, stride)
    range_H = range(0, H - sample_h, stride) if H - sample_h > 0 else range(0, 1)
    range_W = range(0, W - sample_w, stride) if W - sample_w > 0 else range(0, 1)
    total_num = len(range_L) * len(range_H) * len(range_W)
    cut_size_3D=(cut_size, cut_size, cut_size)

    for k in tqdm(range_L):
        for i in range_H:
            for j in range_W:
                # 获取2D图片和3D结构
                sample_num = sample_num + 1
                # print("processing sample %d/%d " % (sample_num, total_num))
                # print("(k,i,j)=(%d,%d,%d)"% (k,i,j))
                # img_2D = img_3D_data[k, i:i + sample_h, j:j + sample_w]
                temp_3D = img_3D_data[k:k + sample_l, i:i + sample_h, j:j + sample_w]
                img_3D = temp_3D.copy()
                # 添加缩放功能，缩放算法采默认用最近邻插值
                if scale is not None and scale != 1.0:
                    img_3D = util.rescale_image_3D(img_3D, scale)

                # img_3D之前的尺寸是cut_size，不等的话要缩放
                if target_size is not None and target_size!=cut_size_3D:
                    img_3D = util.resize_image_3D(img_3D, target_size)

                if transform and np.random.rand(1) > 0.2:
                    img_3D=util.MyRandomTranform()(img_3D)

                # print(img_3D.shape)
                # 2D图像为3D图像的第一张
                img_2D = img_3D[0]
                h5_path = os.path.join(out_dir, "sample_%04d.h5" % sample_num)
                # 保存为h5文件
                with h5py.File(h5_path, 'w') as f:
                    # dtype="i8",表示数据类型为int 8位，compression="gzip"表示压缩方法选择的gzip
                    f.create_dataset('img_2D', data=img_2D, dtype="i8", compression="gzip")
                    f.create_dataset('img_3D', data=img_3D, dtype="i8", compression="gzip")

    print("creating %d samples is finished!" % sample_num)


def Save_img_from_HDF5():
    # file_path = "../datasets/sandstone_2_512/sample_0001.h5"
    file_path = "../datasets/sandstone_256_combined/test/sample_0150.h5"
    with h5py.File(file_path, "r") as f:
        img_2D = f['img_2D'][:]
        img_3D = f['img_3D'][:]
    path_2D = "../datasets/my_test/2D"
    if not os.path.exists(path_2D):
        os.makedirs(path_2D)
    path_3D = "../datasets/my_test/3D"
    if not os.path.exists(path_3D):
        os.makedirs(path_3D)
    img_name = os.path.join(path_2D, "Image000.bmp")
    cv2.imwrite(img_name, img_2D)

    for i in range(len(img_3D)):
        img_name = os.path.join(path_3D, "Image%03d.bmp" % i)
        cv2.imwrite(img_name, img_3D[i])

    # for i in [5,50,70,100,150,170]:
    #     file_path = "../datasets/sandstone_combined/test/sample_%04d.h5" %i
    #     with h5py.File(file_path, "r") as f:
    #         img_2D = f['img_2D'][:]
    #         img_3D = f['img_3D'][:]
    #     path_2D = "../datasets/my_test/img_%04d/2D" %i
    #     if not os.path.exists(path_2D):
    #         os.makedirs(path_2D)
    #     path_3D = "../datasets/my_test/img_%04d/3D"%i
    #     if not os.path.exists(path_3D):
    #         os.makedirs(path_3D)
    #     img_name = os.path.join(path_2D, "Image000.bmp")
    #     cv2.imwrite(img_name, img_2D)
    #
    #     for i in range(len(img_3D)):
    #         img_name = os.path.join(path_3D, "Image%03d.bmp" % i)
    #         cv2.imwrite(img_name, img_3D[i])

def main_for_combining_several_dataset():
    input_path = "../datasets/"
    phase = "sandstone"
    # combined_indexes = [3, 4, 5]
    combined_indexes = [3, 4, 5]

    # 每个样本集选择NUM出来
    NUM = 204
    Combine_Samples_For_Training(input_path, phase, combined_indexes, NUM, dir_name="train")

    NUM = 139
    Combine_Samples_For_Testing(input_path, phase, combined_indexes, NUM, dir_name="test")


def main_for_2D_to_3D(input_dir, out_dir, cut_size=128, scale=None,target_size=None,transform=True):
    create_data_for_2D_to_3D(input_dir, out_dir, cut_size=cut_size, scale=scale, target_size=target_size,transform=transform)

    # 数据集的60%作为训练集
    my_util.Split_samples(out_dir, ratio_for_trainset=0.6)
    print("finished!")


if __name__ == '__main__':

        # 11-28注释掉for c in [4]:
            # opt.CLASS = 'sandstone_%d'%(c)  # choices=['silicon','sandstone','manmade_sandstone','battery','sandstone_low_posority']
            # 11-28注释掉opt.CLASS ='sandstone_%d'%(c)
            opt.CLASS='sandstone_multi'
            opt.input_image_path = os.path.join('../datasets/original_imagesets/',opt.CLASS)
            # opt.output_path = os.path.join('../datasets', opt.CLASS+'_square')
            # opt.output_path = os.path.join('../datasets', opt.CLASS+"_256")
            # opt.ouput_base_path = os.path.join('../datasets', opt.CLASS)
            opt.ouput_base_path = os.path.join('../datasets', opt.CLASS + "_128")
            # opt.ouput_base_path = os.path.join('../datasets', 'Berea_512')
            print(opt)
            # main_for_2D_to_3D(input_dir=opt.input_image_path, out_dir=opt.input_base_path, cut_size=256,scale=0.5)
            # main_for_2D_to_3D(input_dir=opt.input_image_path, out_dir=opt.ouput_base_path, cut_size=128, target_size=(128, 128, 128))

            main_for_2D_to_3D(input_dir=opt.input_image_path, out_dir=opt.ouput_base_path, cut_size=256, target_size=(128,128,128),transform=True)
            # main_for_2D_to_3D(input_dir=opt.input_image_path, out_dir=opt.ouput_base_path, cut_size=256, transform=True)
        # Save_img_from_HDF5()
        #
        # main_for_combining_several_dataset()
#