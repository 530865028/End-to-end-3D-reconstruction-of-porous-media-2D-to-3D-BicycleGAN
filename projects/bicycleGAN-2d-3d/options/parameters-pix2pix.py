#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2018/11/1
-------------------------------------------------
"""
import os
###########################################################
    #training和testing共用的参数
###########################################################


####################bicycle_gan的参数######################

GRAY_THRESHOLD=128
# INDEX=20 # 保留区域的高度

lamb_L1=10 # L1的权重系数
lamb_L1_Harddata=0.0 # 硬数据L1的权重系数

# lamb_Pattern=1*1e3 # 图像修复Pattern_loss的权重系数
# lamb_Pattern=5*1e7 # Pattern_loss的权重系数
lamb_Pattern=0.0 # Pattern_loss的权重系数
lamb_Pattern_multiscale=0.0*lamb_Pattern
lamb_Porosity=5.0*1e5 # 图像修复Pattern_loss的权重系数
# lamb_Porosity=1*1e3 # Pattern_loss的权重系数
# lamb_Porosity=0.0 # Pattern_loss的权重系数

MODEL='bicycle_gan'
# dataset details
CLASS='sandstone_5'  # 数据集文件夹的名字,
# choice=['manmade_sandstone_inpainting','battery_square_26','battery_4_subareas','sandstone','sandstone_20',,'manmade_sandstone_10','battery_20','sandstone_low_posority','sandstone_1_combined']
DATAROOT='./datasets/'+CLASS  # 读取数据集的目录

# FILENAME=CLASS+'_lamb_Pattern_%d_lamb_Porosity_%d' %(lamb_Pattern,lamb_Porosity)
# FILENAME=CLASS+'_lamb_Pattern_%d_lamb_Porosity_%d_lamb_MultiScale_%d' %(lamb_Pattern,lamb_Porosity,lamb_Pattern_multiscale)
# FILENAME=CLASS+'_lamb_L1_%d_lamb_Pattern_%d_lamb_Porosity_%d_lamb_MultiScale_%d' %(lamb_L1_Target,lamb_Pattern,lamb_Porosity,lamb_Pattern_multiscale)
FILENAME=CLASS+'_lamb_L1_%d_lamb_L1_Hardata_%d_lamb_Pattern_%d_lamb_Porosity_%d_lamb_MultiScale_%d' %(lamb_L1,lamb_L1_Harddata,lamb_Pattern,lamb_Porosity,lamb_Pattern_multiscale)
CHECKPOINTS_DIR=os.path.join('./checkpoints/',FILENAME)

NAME=CLASS+'_'+MODEL # checkpoints下的第二层文件夹

# test的时候，输出图片路径
RESULTS_DIR='./results/'+FILENAME
# checkpoint dir
TEST_CHECKPOINT_DIR=CHECKPOINTS_DIR
EPOCH=300  # 数字或者 'latest'

####################bicycle_gan的参数######################



'''
lamb_Pattern=0.0*1e5 # Pattern_loss的权重系数
lamb_Pattern_multiscale=1.0*lamb_Pattern
lamb_Porosity=0.0*1e3 # Pattern_loss的权重系数



####################pix2pix参数######################

GRAY_THRESHOLD=128
lamb_L1=100 # L1的权重系数
MODEL='pix2pix'

# dataset details
CLASS='sampling_image'  # 数据集文件夹的名字,
# choice=['manmade_sandstone_inpainting','battery_square_26','battery_4_subareas','sandstone','sandstone_20',,'manmade_sandstone_10','battery_20','sandstone_low_posority','sandstone_1_combined']
DATAROOT='./datasets/'+CLASS  # 读取数据集的目录

FILENAME=CLASS+'_lamb_L1_%d' %(lamb_L1)
CHECKPOINTS_DIR=os.path.join('./checkpoints/',FILENAME)

# NAME=CLASS+'_'+MODEL # checkpoints下的第二层文件夹
NAME=CLASS+'_'+MODEL+'-large_trainset' # checkpoints下的第二层文件夹

# test的时候，输出图片路径
RESULTS_DIR='./results/'+FILENAME
# checkpoint dir
TEST_CHECKPOINT_DIR=CHECKPOINTS_DIR
EPOCH=400  # 数字或者 'latest'

'''