#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2018/12/27
-------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():
    np.random.seed(123)
    input_img = Image.open('input_004.bmp')
    input_img_RGB = input_img.convert('RGB')
    target_img = Image.open('Rec3.bmp')

    target_img_RGB = target_img.convert('RGB')
    img_np=np.array(input_img)

    print(img_np[0,0])
    # plt.figure(figsize=(1.30, 1.30))

    plt.imshow(input_img_RGB)
    # plt.show()

    #TODO 采样点的功能
    # for i in range()

    n1 = 10
    x1 = np.random.randint(1, 127, n1)  # 平均值为0，方差为1，生成1024个数

    y1 = np.random.randint(30, 127, n1)
    # plt.scatter(x1, y1, s=25,c='r',alpha=0.5)   # s为size，按每个点的坐标绘制，alpha为透明度
    s1=plt.scatter(x1, y1, s=30,c='springgreen')   # s为size，按每个点的坐标绘制，alpha为透明度

    n2 = 40
    x2 = np.random.randint(1, 127, n2)  # 平均值为0，方差为1，生成1024个数
    y2 = np.random.randint(30, 127, n2)

    # plt.scatter(x1, y1, s=25,c='b',alpha=0.5)   # s为size，按每个点的坐标绘制，alpha为透明度
    s2=plt.scatter(x2, y2, s=30,c='darkorange')   # s为size，按每个点的坐标绘制，alpha为透明度


    plt.legend(handles=[s1, s2, ], labels=['pore', 'solid'], loc='best')
    # plt.xlim(0,128)
    # plt.ylim(0,128)
    #
    plt.xticks([140])
    plt.yticks([140])
    #
    plt.savefig('test.png')
    plt.show()

    plt.imshow(target_img_RGB)
    s3=plt.scatter(x1, y1, s=30,c='springgreen')   # s为size，按每个点的坐标绘制，alpha为透明度
    s4=plt.scatter(x2, y2, s=30,c='darkorange')   # s为size，按每个点的坐标绘制，alpha为透明度

    plt.legend(handles=[s3, s4, ], labels=['pore', 'solid'], loc='best')
    # plt.xlim(0,128)
    # plt.ylim(0,128)
    #
    plt.xticks([140])
    plt.yticks([140])
    #
    plt.savefig('test_2.png')
    plt.show()


    for i in range(len(x1)):
        img_np[x1[i]][y1[i]]=255

    for i in range(len(x2)):
        img_np[x2[i]][y2[i]]=0

    final_img=Image.fromarray(np.uint8(img_np))
    final_img.save('HardData.bmp')

def test():
    # 导入必要的模块
    import numpy as np
    import matplotlib.pyplot as plt
    # 产生测试数据
    x = np.arange(1, 10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    s1=ax1.scatter(x, y, c='r', marker='o')
    # 设置图标
    # plt.legend('x1')

    x = np.arange(10, 20)
    y = x

    s2=ax1.scatter(x, y, c='b', marker='o')
    # 设置图标
    plt.legend(handles=[s1, s2,], labels=['a1212', 'b2323'], loc='best')

    # 显示所画的图
    plt.show()

def mark_HardData_on_Rec():
    input_img = Image.open('116_input-harddata.bmp')
    # input_img_RGB = input_img.convert('RGB')
    target_name='116_input-harddata_rec_1.bmp'
    target_img = Image.open(target_name)
    target_img_RGB = target_img.convert('RGB')
    img_np = np.array(input_img)
    target_img_np=np.array(target_img_RGB)
    target_img_np=np.zeros_like(target_img_np)

    H,W=img_np.shape
    for i in range(20,H):
        for j in range(W):
            if img_np[i,j]==255:
                target_img_np[i,j]=(255,140,0)  #橙色
            if img_np[i,j]==0:
                target_img_np[i,j]=(0,255,127) #嫩绿

    final_img = Image.fromarray(np.uint8(target_img_np))
    # final_img.save('mark'+target_name)
    final_img.save('mark.png')


if __name__ == '__main__':
    # test()
    main()
    # mark_HardData_on_Rec()