#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/6/19
-------------------------------------------------
"""

from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
from matplotlib.pyplot import MultipleLocator
TITLE_FONT_SIZE=28
AXIS_FONT_SIZE=24


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.where(x < 0, 0, x)


def LeakyReLU(x):
    return np.where(x < 0, 0.2 * x, x)


def plot_sigmoid():
    x = np.arange(-10, 10, 0.01)
    y = sigmoid(x)
    ax = plt.gca()  # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y,linewidth=3.0)
    plt.title("Sigmoid", fontsize=TITLE_FONT_SIZE)  # 指定标题，并设置标题字体大小

    x_major_locator = MultipleLocator(5)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数

    plt.xlim([-11.05, 11.05])
    plt.ylim([-0.02, 1.12])
    plt.xticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.yticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.tight_layout()
    plt.savefig("Sigmoid.png",dpi=300,bbox_inches = 'tight')
    plt.show()


def plot_tanh():

    x = np.arange(-10, 10, 0.01)
    y = tanh(x)

    ax = plt.gca()  # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y, linewidth=3.0)

    x_major_locator = MultipleLocator(5)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数

    plt.xlim([-11.05, 11.05])
    plt.ylim([-1.12, 1.12])
    plt.xticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.yticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.title("Tanh", fontsize=TITLE_FONT_SIZE)  # 指定标题，并设置标题字体大小
    plt.tight_layout()
    plt.savefig("Tanh.png",dpi=300,bbox_inches = 'tight')
    plt.show()


def plot_relu():

    x = np.arange(-10, 10, 0.01)
    y = relu(x)
    ax = plt.gca()  # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y, linewidth=3.0)

    x_major_locator = MultipleLocator(5)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlim([-11.05, 11.05])
    plt.ylim([0, 11.02])
    plt.xticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.yticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)

    plt.title("ReLU", fontsize=TITLE_FONT_SIZE)  # 指定标题，并设置标题字体大小
    plt.tight_layout()
    plt.savefig("ReLU.png",dpi=300,bbox_inches = 'tight')
    plt.show()


def plot_LeakyReLU():

    x = np.arange(-10, 10, 0.01)
    y = LeakyReLU(x)
    ax = plt.gca()  # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))

    plt.plot(x, y, linewidth=3.0)
    x_major_locator = MultipleLocator(5.0)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(5.0)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlim([-11.05, 11.05])
    plt.ylim([-11.05, 11.05])
    plt.xticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.yticks(fontproperties='Times New Roman', size=AXIS_FONT_SIZE)
    plt.title("LeakyReLU(a=0.2)", fontsize=TITLE_FONT_SIZE)  # 指定标题，并设置标题字体大小
    plt.tight_layout()

    plt.savefig("LeakyReLU.png",dpi=300,bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    plot_sigmoid()
    plot_tanh()
    plot_relu()
    plot_LeakyReLU()
