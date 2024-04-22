#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/3/7
-------------------------------------------------
"""


import numpy as np
import os
import glob
import h5py
import time
import random
import shutil
import cv2
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='create traning set or testing set')
parser.add_argument('--input_path', required=False, type=str,default='',help='path of all correlation function' )
parser.add_argument('--img_total_num', required=False, type=int,default=20,help='path of all correlation function' )

opt = parser.parse_args()



def Select_Desired_Img(input_path,IMG_TOTAL_NUM=20):
    print('当前处理的文件路径为：',input_path)
    dirs=os.listdir(input_path)
    print("当前路径的文件夹为：",dirs)

    Desired_Img_dict={}
    dif_list=[]

    # 文件夹循环
    for i,dir in enumerate(dirs):
       # 对每个文件夹里面图像读取
        if os.path.isdir(os.path.join(input_path,dir)):

            # 保存重建序列图的孔隙度
            porosity_list=[]
            for j in range(IMG_TOTAL_NUM):
                # 读序列图
                rec_img_name="input_%03d_random_sample%02d.bmp" %(i,j+1)

                rec_img_name=os.path.join(input_path,dir,rec_img_name)
                # print(rec_img_name)
                rec_img=Image.open(rec_img_name)
                porosity_list.append(Cal_Porosity(rec_img))

            # 转为numpy
            porosity_recs=np.array(porosity_list)

            # 读取target图像，并计算孔隙度
            target_img_name = "input_%03d_ground truth.bmp" % (i)
            target_img_name = os.path.join(input_path, dir, target_img_name)
            target_img = Image.open(target_img_name)
            porosity_target = np.array([Cal_Porosity(target_img)])

            # 计算重建图的孔隙度均值与目标值的差异
            dif=np.abs(np.mean(porosity_recs)-porosity_target)
            # dif=np.square(np.mean(porosity_recs)-porosity_target)
            # 取出数值
            dif=np.around(dif, decimals=8)
            dif_list.append(dif[0])

            # np.sort(np.array(Desired_Img_Index))

            #保存图像孔隙度数据为txt文件
            FileNames = ["porosity-%d.txt" % (IMG_TOTAL_NUM),"porosity-target.txt"]
            Statistics=[porosity_recs,porosity_target]
            for FileName, data in zip(FileNames, Statistics):
                FilePath = os.path.join(input_path,dir,FileName)
                np.savetxt(FilePath, data, fmt='%0.8f')

    Desired_Img_dict=dict(zip(dif_list,dirs))
    #按照key的升序来排序
    Desired_Img_dict= dict(sorted(Desired_Img_dict.items(), key=lambda d:d[0],reverse=False))
    #print(Desired_Img_dict)

    path=os.path.join(input_path,'重建图的平均孔隙度与目标值的差异.txt')
    f = open(path, 'w')
    f.write('绝对值差异:\t文件夹名字\n')
    for key,value in Desired_Img_dict.items():
        f.write(str(key) + '\t' + str(value))
        f.write('\n')
    f.close()


def Cal_Porosity(img):
    img=np.array(img)
    img=img>0
    num_porosity = (img != 0).sum()
    w,h=img.shape
    return num_porosity/(w*h)


if __name__ == '__main__':
    opt.input_path="D:/PyCharm_Project/Pytorch_Test/BicycleGAN/results/sandstone_20_lamb_Pattern_0_lamb_Porosity_0/test_epoch_200/images"
    # opt.input_path=""
    Select_Desired_Img(opt.input_path,opt.img_total_num)
    print("finished!")
