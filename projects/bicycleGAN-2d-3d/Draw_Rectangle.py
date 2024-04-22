#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/1/15
-------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import os

import argparse
parser = argparse.ArgumentParser(description='create traning set or testing set')

parser.add_argument('--path', required=False, default='',help='path of all correlation function' )
opt = parser.parse_args()


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

ORANGE_COLOR=(225, 85, 1)
BLUE_COLOR=(30,115,215)


def Get_ALL_ImagePaths(img_dir):
    images = []
    assert os.path.isdir(img_dir), '%s is not a valid directory' % img_dir
    files=os.listdir(img_dir)
    for file in files:
        if is_image_file(file):
            path = os.path.join(img_dir, file)
            images.append(path)

    return  images



# 最基本的画矩形的函数

def Basic_Draw_One_Rectangle(img_dir,x,y,width,height,line_width=3):
    img_paths = Get_ALL_ImagePaths(img_dir)
    if len(img_paths) == 0:
        raise RuntimeError('There in no images in dir %s' % img_dir)

    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.convert('RGB')
        w, h = img.size

        # 创建newImg，用来在外面增加一圈，方便画矩形框
        newImg = Image.new("RGB", (w + line_width, h + line_width), (255, 255, 255))
        box = (line_width - 1, line_width - 1)
        newImg.paste(img, box)
        print(newImg.size)
        draw = ImageDraw.Draw(newImg)
        # 原图矩形框为13*13

        # 画一个矩形
        draw.rectangle((x, y, x + width, y + height), fill=None, outline=ORANGE_COLOR, width=line_width)
        img_name = os.path.basename(img_path)
        shortname, extension = os.path.splitext(img_name)
        path = os.path.join(img_dir, 'Rectangle')
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, shortname + "_Rectangle.bmp")
        newImg.save(save_path)

# 画一个方形
def Draw_One_Square(img_dir):
    x, y = (0, 0)
    width, height = (26, 26)
    # 外面的矩形框的边长为两者之和
    line_width = 3
    width, height = (width + line_width + 1, height + line_width + 1)
    Basic_Draw_One_Rectangle(img_dir,x,y,width,height,line_width)

# 画一个矩形
def Draw_One_Rectangle(img_dir):
    x, y = (0, 0)
    width, height = (128, 10)
    # 外面的矩形框的边长为两者之和
    line_width = 3
    width, height = (width + line_width + 1, height + line_width + 1)
    Basic_Draw_One_Rectangle(img_dir, x, y, width, height, line_width)

# 画4个方形
def Draw_Four_Squares(img_dir):
    img_paths = Get_ALL_ImagePaths(img_dir)
    if len(img_paths) == 0:
        raise RuntimeError('There in no images in dir %s' % img_dir)
    # 原图矩形框为13*13
    line_width=3
    width,height=(13,13)
    # 外面的矩形框的边长为两者之和
    width, height = (width+line_width+2, height+line_width+2)
    # 与训练集的坐标相同
    coordinates = [(35, 35), (35, 80), (80, 35), (80, 80)]

    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        for (x, y) in coordinates:
            # 画一个矩形
            x=x-line_width
            y=y-line_width
            draw.rectangle((x, y, x+width, y+height), fill=None, outline=ORANGE_COLOR, width=line_width)
            img_name = os.path.basename(img_path)
            shortname, extension = os.path.splitext(img_name)
            path=os.path.join(img_dir,'Rectangle')
            if not os.path.exists(path):
                os.makedirs(path)
            save_path = os.path.join(path, shortname + "_Rectangle.bmp")
            img.save(save_path)

if __name__ == '__main__':
    # main()

    # path='F:/博士学习/我的专利和论文/论文/第三篇/论文中的图/anisotropic/10/第四组/input_target_rec'
    # Draw_One_Rectangle(path)
    # path = 'F:/博士学习/我的专利和论文/论文/第三篇/论文中的图/有限信息重建完成信息示意图/sandstone'
    # Draw_One_Rectangle(path)
    opt.path='F:/博士学习/我的专利和论文/论文/第三篇/论文中的图/silica/input_target_rec'
    Draw_One_Square(opt.path)
    print('finished!')


