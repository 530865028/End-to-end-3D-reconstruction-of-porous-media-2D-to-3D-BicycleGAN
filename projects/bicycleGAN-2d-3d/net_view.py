#coding=utf-8
"""
-------------------------------------------------
   Description :
   Author :       feng
   date：         2019/4/9
-------------------------------------------------
"""
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#import hiddenlayer as hl
#
import torchvision.models
from graphviz import Digraph
from torch.autograd import Variable
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# 网络模型绘图，生成pytorch autograd图表示，蓝色节点表示要求grad梯度的变量，黄色表示在反向传播的张量
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var_grad):
        if var_grad not in seen:
            if torch.is_tensor(var_grad):
                dot.node(str(id(var_grad)), size_to_str(var_grad.size()), fillcolor='orange')
            elif hasattr(var_grad, 'variable'):
                u = var_grad.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var_grad)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var_grad)), str(type(var_grad).__name__))
            seen.add(var_grad)
            if hasattr(var_grad, 'next_functions'):
                for u in var_grad.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var_grad)))
                        add_nodes(u[0])
            if hasattr(var_grad, 'saved_tensors'):
                for t in var_grad.saved_tensors:
                    dot.edge(str(id(t)), str(id(var_grad)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def tensorboard_demo_2():
    # 指定训练时的参数
    opt = TrainOptions().parse()
    # 创建 dataloader
    data_loader = CreateDataLoader(opt)
    # 加载数据
    dataset = data_loader.load_data()
    # 得到训练样本的个数
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # 创建模型(bicycle_gan_model'或者'pix2pix_model)
    model = create_model(opt)
    # 设置是开始训练还是加载网络来进行测试
    model.setup(opt)
    model_G = model.netD.cpu()
    print(model_G)
    z = torch.rand(1, 8, 1, 1)
    x = torch.rand(1, 1, 128, 128)
    # hl.build_graph(model_E, (x,z))
    # y=model_E(x,z)
    with SummaryWriter(comment='_netD') as w:
        w.add_graph(model_G, x)

def tensorboard_demo():

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Sequential(  # input_size=(1*28*28)
                nn.Conv2d(1, 6, 5, 1, 2),
                nn.ReLU(),  # (6*28*28)
                nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),  # (16*10*10)
                nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU()
            )
            self.fc3 = nn.Linear(84, 10)

        # 定义前向传播过程，输入为x
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
            x = x.view(x.size()[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    dummy_input = torch.rand(13, 1, 28, 28)  # 假设输入13张1*28*28的图片
    model = LeNet()
    with SummaryWriter(comment='_LeNet') as w:
        w.add_graph(model, (dummy_input,))


def main():

    #指定训练时的参数
    opt = TrainOptions().parse()
    # 创建 dataloader
    data_loader = CreateDataLoader(opt)
    # 加载数据
    dataset = data_loader.load_data()
    # 得到训练样本的个数
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # 创建模型(bicycle_gan_model'或者'pix2pix_model)
    model = create_model(opt)
    # 设置是开始训练还是加载网络来进行测试
    model.setup(opt)
    model_E=model.netE.cpu()
    print(model_E)
    z=torch.rand(1, 8,1, 1)
    x=torch.rand(1, 1, 128, 128)
    # hl.build_graph(model_E, (x,z))
    #y=model_E(x,z)
    with SummaryWriter(comment='_netG') as w:
        w.add_graph(model_E, (x))

    # g = make_dot(y)
    # #g.view()
    # g.render('vis_model_G', view=True)
    # VGG16 with BatchNorm
    # model = torchvision.models.vgg16()
    #
    # # Build HiddenLayer graph
    # # Jupyter Notebook renders it automatically
    # hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
    #

if __name__ == '__main__':
    tensorboard_demo_2()