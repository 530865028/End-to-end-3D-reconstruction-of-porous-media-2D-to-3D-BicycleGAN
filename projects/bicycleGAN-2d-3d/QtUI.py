#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2018/11/20
-------------------------------------------------
"""

#!/usr/bin/python3



from __future__ import print_function
import argparse
import os
import time
import torch
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2

from my_util import numpy_float_to_image

import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images

from options.parameters import EPOCH

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

GRAY_THRESHOLD=10

# options
opt = TestOptions().parse()
opt.gpu_ids=''
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

'''
PyQt5 文件对话框
souwiki.com
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap,QImage

torch.backends.cudnn.deterministic = True
seed=123
torch.manual_seed(seed)

# model_path = 'checkpoint/output_20_lamb_5_lambPattern_200000_lambPorosity_5000_noise_4096_ngf_32_L1_all_MSE/netG_model_epoch_300.pth'

class DemoForImageReconstruction(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Demo for image reconstruction'
        self.left = 100
        self.top = 100
        self.width = 540
        self.height = 380
        self.image_w=128
        self.image_h=128
        self.noise_value=0.0
        self.porosity_value=0.0
        #noise z
        self.z=torch.randn(1,opt.nz)

        self.QIm=QImage()
        self.InputImage=None
        self.RecImage=None

        self.initUI()
        # self.set_noise_value(self.noise_value)

    # def set_noise_value(self,value):
    #
    #     self.z = torch.zeros(1, 1, 1, 1)
    #     self.z[0][0][0][0] = value
    #     self.z = self.z.expand(1, NOISE_SIZE, 1, 1)  # 扩充维度，即沿着dim=1方向，将z拷贝NOISE_SIZE次。

    def initUI(self):

        self.setWindowTitle(self.title)
        self.resize(self.width, self.height)
        self.center()
        # self.setGeometry(self.left, self.top, self.width, self.height)

        # Create widget

        # pixmap = QPixmap('images/TI.bmp')

        # image 1 and btn 1
        self.image_1 = QLabel(self)
        self.image_1.setGeometry(20, 20, self.image_w, self.image_h)
        pixmap_1 = QPixmap()
        self.image_1.setPixmap(pixmap_1)
        # self.label.move(20, 20)


        self.btn_1 = QPushButton('Load image', self)
        self.btn_1.clicked.connect(self.LoadImage)
        self.btn_1.move(40, 200)


        #image 2 and btn 2
        self.image_2 = QLabel(self)
        self.image_2.setGeometry(180, 20, self.image_w, self.image_h)
        pixmap_2 = QPixmap()
        self.image_2.setPixmap(pixmap_2)

        self.btn_2 = QPushButton('rand reconstruct', self)
        self.btn_2.clicked.connect(self.Reconstruct)
        self.btn_2.move(210, 200)

        #btn 3 for image save

        self.btn_3 = QPushButton('save image', self)
        self.btn_3.clicked.connect(self.ImageSave)
        self.btn_3.move(210, 250)

        ##################滑动条#####################

        # self.sld = QSlider(Qt.Horizontal, self)
        # self.sld.setMinimum(0)
        # self.sld.setMaximum(99)
        # self.sld.setValue(50)
        # self.sld.setFocusPolicy(Qt.NoFocus)
        # #设置滑动条的位置，2代表下方（TicksBelow = 2）
        # self.sld.setTickPosition(2)
        # self.sld.setTickInterval(10)
        # self.sld.setGeometry(350,80, 100, 30)
        # self.sld.valueChanged[int].connect(self.ChangeNoise)

        ##################滑动条#####################

        # text for input
        self.text_input = "input"
        self.input_text_lable = QLabel(self.text_input, self)
        # self.text_lable.move(350,120)
        self.input_text_lable.setGeometry(53, 170, 100,20)


        # text for input
        self.text_output = "output"
        self.output_text_lable = QLabel(self.text_output, self)
        # self.text_lable.move(350,120)
        self.output_text_lable.setGeometry(220, 170, 100, 20)


        # noised 的text显示
        # self.noise_text = "noise: {}".format(self.noise_value)
        # self.noise_text_lable = QLabel(self.noise_text, self)
        # # self.text_lable.move(350,120)
        # self.noise_text_lable.setGeometry(350, 120, 200, 80)


        # 孔隙度 的text显示
        self.porosity_text = "porosity: {}".format(self.porosity_value)
        self.porosity_text_lable = QLabel(self.porosity_text, self)
        # self.text_lable.move(350,120)
        self.porosity_text_lable.setGeometry(350, 140, 200, 80)

        # self.resize(pixmap.width(),pixmap.height())

        # self.show()

    def LoadImage(self):
        print("load image")

        # fname, ok = QFileDialog.getOpenFileName(self, 'Select image', 'c:/', 'Image files(*.bmp *.jpg *.png)')
        # fname, ok = QFileDialog.getOpenFileName(self, 'Select image', './', 'Image files(*.bmp *.jpg *.png)')
        fname, ok = QFileDialog.getOpenFileName(self, 'Select image', 'C:/Users/Administrator/Desktop/Test', 'Image files(*.bmp *.jpg *.png)')

        if ok:

            Qt_img=QPixmap(fname)

            shape=(Qt_img.height(),Qt_img.width())
            print(shape)
            # 如果shape不是(128,128)
            if not shape == (128, 128):
                QMessageBox.information(self, 'Message',
                                        "The size of image is not (128,128)", QMessageBox.Yes |
                                        QMessageBox.Cancel, QMessageBox.Yes)
                return

            self.image_1.setPixmap(Qt_img)

            img=cv2.imread(fname,flags=cv2.IMREAD_GRAYSCALE)

            # 转成四个维度[Batch,C,H,W]
            # print(img.shape)

            tensor_img= torch.from_numpy(img).unsqueeze_(0).unsqueeze_(0)

            tensor_img=tensor_img.float().div_(255).sub_(0.5).div_(0.5)  # 先转成FloatTensor，再将原来[0-255]之间的数据转为[-1，-1]的数据
            self.InputImage=tensor_img
            # print(self.InputImage.shape)

    def Reconstruct(self):

        if self.InputImage is None:
            QMessageBox.information(self, 'Message',
                                    "Please load image first!", QMessageBox.Yes |
                                    QMessageBox.Cancel, QMessageBox.Yes)
            return

        # 图片前传。验证和测试时不需要更新网络权重，所以使用torch.no_grad()，表示不计算梯度

        # if opt.cuda:
        #     self.InputImage=self.InputImage.cuda()
        #     self.z=self.z.cuda()

        self.z=torch.randn(1,opt.nz)
        model.set_single_input(self.InputImage)
        with torch.no_grad():
            out = model.test_for_output(self.z)

        out_img = out.data[0].numpy()  # 取出第一个batch的三个维度的数据， 从[batch_size,1,128,128]取出[1,128,128]
        out_img = np.squeeze(out_img, axis=0)  # 去掉第一个维度（单通道）

        out_img = numpy_float_to_image(out_img)
        out_img = out_img > GRAY_THRESHOLD  # out_img的值为True或者False
        out_img = (out_img + 0).astype(np.uint8) * 255  # out_img+0将True或者False转为1或者0

        self.RecImage=out_img

        Im=self.RecImage

        # Im = cv2.imread('C:/Users/Administrator/Desktop/Test/000_output.bmp',cv2.IMREAD_GRAYSCALE)  # 通过Opencv读入一张图片

        image_height, image_width= Im.shape  # 获取图像的高，宽。

        # QIm = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下

        self.QIm = QImage(Im.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                     QImage.Format_Grayscale8)  #8位图

        # print(self.QIm.isNull())

        self.image_2.setPixmap(QPixmap.fromImage(self.QIm))  # 将QImage显示在之前创建的QLabel控件中
        self.Updata_Porosity_Text()

    def ImageSave(self):

        #判断图像是否为空！！！
        if self.QIm.isNull():

            QMessageBox.information(self, 'Message',
                                         "The reconstruction image is NULL!", QMessageBox.Yes |
                                         QMessageBox.Cancel, QMessageBox.Yes)
            return

        # ImgName, ok = QFileDialog.getSaveFileName(self, 'Save', 'c:/', 'BMP (*.bmp) ;; PNG(*.png)')
        # ImgName, ok = QFileDialog.getSaveFileName(self, 'Save', './', 'BMP (*.bmp) ;; PNG(*.png)')
        ImgName, ok = QFileDialog.getSaveFileName(self, 'Save', 'C:/Users/Administrator/Desktop/Test', 'BMP (*.bmp) ;; PNG(*.png)')

        self.QIm.save(ImgName)

    # def ChangeNoise(self,value):
    #
    #     # print(value)
    #     if self.InputImage is None:
    #         QMessageBox.information(self, 'Message',
    #                                 "Please load image first!", QMessageBox.Yes |
    #                                 QMessageBox.Cancel, QMessageBox.Yes)
    #         return
    #
    #     self.noise_value=(value/99.0-0.5)/0.5
    #     self.Reconstruct()
    #     self.Update_Noise_Text()
    #     self.Updata_Porosity_Text()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def GetInputImage(self):

        return self.InputImage

    def SetRecImage(self,CvRecImg):

        self.RecImage=CvRecImg

    # def Update_Noise_Text(self):
    #
    #     text = "noise: %.3f" %(self.noise_value)
    #     self.noise_text_lable.setText(text)

    def Updata_Porosity_Text(self):

        self.Calcu_Posority()
        text = "porosity: %.3f" % (self.porosity_value)
        self. porosity_text_lable.setText(text)

    def Calcu_Posority(self):

        self.porosity_value=self.RecImage.sum()/(255*self.image_w*self.image_h)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = DemoForImageReconstruction()

    # Im = cv2.imread('C:/Users/Administrator/Desktop/Test/000_output.bmp', cv2.IMREAD_GRAYSCALE)  # 通过Opencv读入一张图片

    # ex.SetRecImage(Im)

    ex.show()
    sys.exit(app.exec_())

