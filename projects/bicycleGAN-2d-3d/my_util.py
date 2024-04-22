from __future__ import print_function
import numpy as np
from PIL import Image
import datetime
import os
import cv2
import glob
import shutil
import random
import copy

GRAY_THRESHOLD=10

from tqdm import tqdm

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

import torch


# def load_img(filepath):
#     img = Image.open(filepath).convert('RGB')
#     img = img.resize((256, 256), Image.BICUBIC)
#     return img
#
#
# def save_img(image_tensor, filename):
#     image_numpy = image_tensor.float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     image_numpy = image_numpy.astype(np.uint8)
#     image_pil = Image.fromarray(image_numpy)
#     image_pil.save(filename)
#     print("Image saved as {}".format(filename))

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])



def save_batch_image(image,order_in_dataset,output_path,image_name='output',segment=False):

    #shape of image is：BATCH_SIZE,1,H,W

    N,C,H,W=image.shape
    for i in range(N):
        img=image[i].data[0].numpy()
        img=numpy_float_to_image(img)#转成0-255
        #分割一下
        if segment== True:
            out_img = img > GRAY_THRESHOLD

            out_img = (out_img + 0).astype(np.uint8) * 255
        else:
            out_img=img

        Final_name = output_path + "/%03d_%s.bmp" % (N*order_in_dataset+i,image_name)
        #print('----Final_name=',Final_name)

        cv2.imwrite(Final_name, out_img)


#numpy的浮点数转为图像
def numpy_float_to_image(data):
    return ((data * 0.5 + 0.5) * 255+0.5).astype(np.uint8)


def Image_to_Tensor_normalized(input):
    input=torch.Tensor(input)
    input = input.div(255).sub(0.5).div(0.5)  # 原图像为[0-255]之间的数据转为[-1，-1]的数据

    return input

def Convert_to_0_255(input):

    return input.mul(0.5).add(0.5).mul(255)


def save_to_txt(arg_name_space,output_path,name):
    argsVars=vars(arg_name_space)
    if not os.path.exists(output_path):
       os.makedirs(output_path)

    with open(output_path+"/"+name, 'w') as f:  # 打开test.txt   如果文件不存在，创建该文件。
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        f.write("log time:"+now_time+"\n\n")
        f.write("parameters information is given as follow, and the format is (key,value)\n\n")
        for (key, value) in argsVars.items():
            f.write(key + ':' + str(value)+"\n")


#将图片放入网页中显示,2018-11-01
def create_index_html(output_path,Lenght_of_Image):
    index_path = os.path.join(output_path, "index.html")
    #print(index_path)
    # if os.path.exists(index_path):
    #     index = open(index_path, "a")#追加
    #else:

    index = open(index_path, "w")#创建新的
    index.write("<html><body><table><tr>")
    index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for i in range(Lenght_of_Image):
        index.write("<tr>")

        index.write("<td>Image%03d</td>" % i)

        for kind in ["input", "output", "target"]:
            index.write("<td><img src='images/%03d_%s.bmp'></td>" % (i,kind))

        index.write("</tr>")
    print('index.html is saved at:',index_path)


#移动文件,2018-11-02

def MyMoveFile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))

#拷贝文件,2018-11-02
def MyCopyFile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))


#srcfile,srcfile均为list,2018-11-02
def BatchCopyFile(srcfile,dstfile):
    if(len(srcfile)!=len(dstfile)):
        raise RuntimeError("the length of srcfile and dstfile must be the same!")

    for i in tqdm(range(len(srcfile))):
        MyCopyFile(srcfile[i],dstfile[i])

    print("Copying files is finished")

#封装成函数。2018-11-02
def SplitImages():

    ratio=0.3
    NUM=None
    Length_of_Training=0
    Length_of_Testing=0

    for i in range(1,14):
        print("------processing %d data set------" % (i))
        image_path='original_imagesets/all_images/%d' %(i)
        if not os.path.exists(image_path):
            raise RuntimeError(('path %s do not exit')%(image_path))

        all_image_path = glob.glob(os.path.join(image_path, "*.bmp"))  # 得到当前文件夹下的所有bmp格式的图片

        random.seed(123)#设置种子点，保证每次的shuffle是一致的
        random.shuffle(all_image_path)  #in-place替换all_image_path

        #处理训练集
        NUM=int(len(all_image_path)*ratio)

        training_image_path=all_image_path[:NUM]  #取前半部分作为训练集

        for j in range(len(training_image_path)):
            output_training_image_path = 'original_imagesets/all_images/images/training2/Image%04d.bmp' % (Length_of_Training+ j)
            MyCopyFile(training_image_path[j],output_training_image_path)

        Length_of_Training += len(training_image_path)#为了后续更新标号


        # 处理测试集
        testing_image_path = all_image_path[NUM:] #取后半部分作为训练集
        for j in range(len(testing_image_path)):
            output_testing_image_path = 'original_imagesets/all_images/images/testing2/Image%04d.bmp' % (
                    Length_of_Testing + j)
            MyCopyFile(testing_image_path[j], output_testing_image_path)

        Length_of_Testing += len(testing_image_path) #为了后续更新标号

def SplitImages_For_Silicon():

    ratio=0.3
    NUM=None
    Length_of_Training=0
    Length_of_Testing=0

    for i in range(1,14):
        print("------processing %d data set------" % (i))
        image_path='original_imagesets/all_images/%d' %(i)
        if not os.path.exists(image_path):
            raise RuntimeError(('path %s do not exit')%(image_path))

        all_image_path = glob.glob(os.path.join(image_path, "*.bmp"))  # 得到当前文件夹下的所有bmp格式的图片

        random.seed(123)#设置种子点，保证每次的shuffle是一致的
        random.shuffle(all_image_path)  #in-place替换all_image_path

        #处理训练集
        NUM=int(len(all_image_path)*ratio)

        training_image_path=all_image_path[:NUM]  #取前半部分作为训练集

        for j in range(len(training_image_path)):
            output_training_image_path = 'original_imagesets/all_images/images/training2/Image%04d.bmp' % (Length_of_Training+ j)
            MyCopyFile(training_image_path[j],output_training_image_path)

        Length_of_Training += len(training_image_path)#为了后续更新标号


        # 处理测试集
        testing_image_path = all_image_path[NUM:] #取后半部分作为训练集
        for j in range(len(testing_image_path)):
            output_testing_image_path = 'original_imagesets/all_images/images/testing2/Image%04d.bmp' % (
                    Length_of_Testing + j)
            MyCopyFile(testing_image_path[j], output_testing_image_path)

        Length_of_Testing += len(testing_image_path) #为了后续更新标号


def Split_samples(sample_dir,ratio_for_trainset):
    all_image_path = [x for x in sorted(os.listdir(sample_dir)) if is_hdf5_file(x)]

    print("样本个数为：", len(all_image_path))

    if ratio_for_trainset > 1.0:
        raise ValueError("parameter ratio_for_trainset should be no more than 1.0")

    shuflled_img_path = copy.deepcopy(all_image_path)
    random.seed(123)  # 设置种子点，保证每次的shuffle是一致的
    random.shuffle(shuflled_img_path)  # in-place替换all_image_path

    #############################train#####################################
    num_for_trainset = int(ratio_for_trainset * len(shuflled_img_path))

    if num_for_trainset % 2 != 0:
        num_for_trainset = num_for_trainset - 1

    for i, img_path in enumerate(shuflled_img_path[:num_for_trainset]):
        in_path = os.path.join(sample_dir, img_path)
        out_dir = os.path.join(sample_dir, "train")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, all_image_path[i])
        MyCopyFile(in_path, out_path)

    #############################train#####################################

    #############################test#####################################

    for i, img_path in enumerate(shuflled_img_path[num_for_trainset:]):
        in_path = os.path.join(sample_dir, img_path)
        out_dir = os.path.join(sample_dir, "test")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, all_image_path[i])
        MyCopyFile(in_path, out_path)

    #############################test#####################################


if __name__ == '__main__':
    # SplitImages()
    Split_samples("datasets/manmade_sandstone", 0.8)
