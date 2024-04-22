#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/1/7
-------------------------------------------------
"""

import numpy as np
import os
import glob
import h5py
import time
import random
import shutil

import argparse
parser = argparse.ArgumentParser(description='create traning set or testing set')

parser.add_argument('--input_path', required=False, default='',help='path of all correlation function' )

opt = parser.parse_args()




# 将相关函数组合为一个txt保存
def Combine_all_Average_Correlation_From_txt(input_path):
    print('当前处理的文件路径为：',input_path)
    dirs=os.listdir(input_path)
    # print(dirs)

    Avg_correlations=[]
    Avg_Lins=[]
    Avg_Clusters=[]
    Statistics=[Avg_correlations,Avg_Lins,Avg_Clusters]
    files=['Average_correlation.txt','Average_Lin.txt','Average_Cluster.txt']
    for dir in dirs:
        # 判断是否是文件
        # os.path.isfile(dir):
        # 判断是否是文件夹
        if os.path.isdir(os.path.join(input_path,dir)):
            for i,file in enumerate(files):
                file_path=os.path.join(input_path,dir,file)
                temp=np.loadtxt(file_path, np.float32)
                # 依次添加进去
                Statistics[i].append(temp)

    # 依次处理
    for i in range(len(Statistics)):
        Statistics[i]=MyTransform(Statistics[i])

    FileNames=['Avg_correlations.txt','Avg_Lins.txt','Avg_Clusters.txt']
    for FileName,data in zip(FileNames,Statistics):
        FilePath=os.path.join(input_path,FileName)
        np.savetxt(FilePath, data, fmt='%0.8f')

    # Avg_correlations=MyTransform(Avg_correlations)
    # Avg_Lins = MyTransform(Avg_Lins)
    # Avg_Clusters = MyTransform(Avg_Clusters)
    # datas=[Avg_correlations,Avg_Lins,Avg_Clusters]


# 将相关函数组合为一个txt保存
def Combine_all_Correlation_35_135_From_txt(input_path):
    print('当前处理的文件路径为：',input_path)
    dirs=os.listdir(input_path)
    # print(dirs)

    Correlations_45=[]
    Correlations_135=[]
    Lins_45=[]
    Lins_135=[]
    Clusters_45=[]
    Clusters_135=[]

    Statistics=[Correlations_45,Correlations_135,Lins_45,Lins_135,Clusters_45,Clusters_135]

    files=['Correlation_45.txt','Correlation_135.txt','Lineal_45.txt','Lineal_135.txt','Cluster_45.txt','Cluster_135.txt']
    for dir in dirs:
        # 判断是否是文件
        # os.path.isfile(dir):
        # 判断是否是文件夹
        if os.path.isdir(os.path.join(input_path,dir)):
            for i,file in enumerate(files):
                file_path=os.path.join(input_path,dir,file)
                temp=np.loadtxt(file_path, np.float32)
                #依次append进去
                Statistics[i].append(temp)

    for i in range(len(Statistics)):
        Statistics[i] = MyTransform(Statistics[i])

    FileNames = ['Correlations_45.txt','Correlations_135.txt','Lineals_45.txt','Lineals_135.txt','Clusters_45.txt','Clusters_135.txt']
    for FileName, data in zip(FileNames, Statistics):
        FilePath = os.path.join(input_path, FileName)
        np.savetxt(FilePath, data, fmt='%0.8f')

#功能:将M个N*1的向量，转为N*M格式

def MyTransform(input):

    if input is None:
        raise ValueError
    input=np.array(input)
    height, width=input.shape
    output=input[0]
    # 增加一个维度
    output=output[:,None]
    # print(output.shape)
    # 下标从1开始
    for i in range(1,height):
        temp = input[i][:,None]
        output=np.concatenate((output,temp),axis=1)

    return output

if __name__ == '__main__':
    # opt.input_path='F:/博士学习/我的专利和论文/论文/第三篇/论文中的图/battery2/第一组/img_002/统计参数'
    opt.input_path='F:/博士学习/我的专利和论文/论文/第三篇/论文中的图/约束函数对比/3两者共同约束/img_017/统计参数'
    Combine_all_Average_Correlation_From_txt(opt.input_path)
    # Combine_all_Correlation_35_135_From_txt(opt.input_path)
    print('Finished')

