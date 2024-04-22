# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable
###############################################################################
# Functions
###############################################################################
# 分割的阈值GRAY_THRESHOLD设为128：y=1 if x>=128 else 0

from options.parameters  import GRAY_THRESHOLD

# 网络参数初始化
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# 网络初始化（并行+权重初始化）
def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    init_weights(net, init_type)
    return net



# opt.lr_policy 默认值为'lambda'
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# 归一化方法
def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer

# 激活函数
# functools.partial作用于或返回其他函数的函数。直白地说，提前传入函数部分的参数，返回该函数。
def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


# 定义G
def define_G(input_nc, output_nc, nz, ngf,
             netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', gpu_ids=[], where_add='input', upsample='bilinear',add_conv=False,noise_expand=True):
    net = None
    # 归一化方法
    norm_layer = get_norm_layer(layer_type=norm)
    # 激活函数
    nl_layer = get_non_linearity(layer_type=nl)

    if nz == 0:
        where_add = 'input'
    # 定义了四种G网络，实质上按照where_add来区分，只有两种，层数不同而已。
    if netG == 'unet_128' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample,add_conv=add_conv,noise_expand=noise_expand)
    elif netG == 'unet_256' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample,add_conv=add_conv,noise_expand=noise_expand)
    elif netG == 'unet_512' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, nz, 9, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample,add_conv=add_conv,noise_expand=noise_expand)
    elif netG == 'unet_128' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample,add_conv=add_conv)
    elif netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample,add_conv=add_conv)
    elif netG == 'unet_512' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 9, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample, add_conv=add_conv)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    #
    return init_net(net, init_type, gpu_ids)


def define_D(input_nc, ndf, netD,
             norm='batch', nl='lrelu',
             use_sigmoid=False, init_type='xavier', num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_128':
    # D_NLayers 就是一般的卷积+激活
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                        nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_256':
        net = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
                        nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif netD == 'basic_128_multi':
    # 卷积+Norm+激活,默认是这个
        net = D_NLayersMulti_2D(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer,
                                use_sigmoid=use_sigmoid, num_D=num_Ds)
    elif netD == 'basic_256_multi':
        net = D_NLayersMulti_2D(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer,
                                use_sigmoid=use_sigmoid, num_D=num_Ds)
    elif netD == 'basic_512_multi':
        net = D_NLayersMulti_2D(input_nc=input_nc, ndf=ndf, n_layers=4, norm_layer=norm_layer,
                            use_sigmoid=use_sigmoid, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


def define_E(input_nc, output_nc, ndf, netE,
             norm='batch', nl='lrelu',
             init_type='xavier', gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    # vaeLike在外面赋值为True
    if netE == 'resnet_128':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_512':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=6, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_512':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=6, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


# class ListModule(object):
#     # should work with all kind of module
#     def __init__(self, module, prefix, *args):
#         self.module = module
#         self.prefix = prefix
#         self.num_module = 0
#         for new_module in args:
#             self.append(new_module)
#
#     def append(self, new_module):
#         if not isinstance(new_module, nn.Module):
#             raise ValueError('Not a Module')
#         else:
#             self.module.add_module(self.prefix + str(self.num_module), new_module)
#             self.num_module += 1
#
#     def __len__(self):
#         return self.num_module
#
#     def __getitem__(self, i):
#         if i < 0 or i >= self.num_module:
#             raise IndexError('Out of bound')
#         return getattr(self.module, self.prefix + str(i))


class D_NLayersMulti_2D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=1):
        super(D_NLayersMulti_2D, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
        # get_layers，代表卷积+Norm+激活，一个get_layers相当于一个D。
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:

            # 添加第一个D
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)

            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            # 继续添加D，即有两个D

            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2 ** i)))

                layers = self.get_layers(
                    input_nc, ndf_i, n_layers, norm_layer, use_sigmoid)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    # 实质上是定义鉴别器D
    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        kw = 4
        padw = 1
        print("input_nc:",input_nc)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        # n_layers在option里面设置为2
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            # self.model是列表，里面是num_D个D。这里是2个。
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            # 如果i!=1，即i=0时，赋值down为AvgPool2d的结果，再作为i=1时的输入。
            if i != self.num_D - 1:
                down = self.down(down)
        return result


# Defines the conv discriminator with the specified arguments.
class G_NLayers(nn.Module):
    def __init__(self, output_nc=3, nz=100, ngf=64, n_layers=3,
                 norm_layer=None, nl_layer=None):
        super(G_NLayers, self).__init__()

        kw, s, padw = 4, 2, 1
        sequence = [nn.ConvTranspose2d(
            nz, ngf * 4, kernel_size=kw, stride=1, padding=0, bias=True)]
        if norm_layer is not None:
            sequence += [norm_layer(ngf * 4)]

        sequence += [nl_layer()]

        nf_mult = 4
        nf_mult_prev = 4
        for n in range(n_layers, 0, -1):
            nf_mult_prev = nf_mult
            nf_mult = min(n, 4)
            sequence += [nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult,
                                            kernel_size=kw, stride=s, padding=padw, bias=True)]
            if norm_layer is not None:
                sequence += [norm_layer(ngf * nf_mult)]
            sequence += [nl_layer()]

        sequence += [nn.ConvTranspose2d(ngf, output_nc,
                                        kernel_size=4, stride=s, padding=padw, bias=True)]
        sequence += [nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class D_NLayers(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, use_sigmoid=False):
        super(D_NLayers, self).__init__()

        kw, padw, use_bias = 4, 1, True
        # st()
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw,
                      stride=2, padding=padw, bias=use_bias),
            nl_layer()
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4,
                               stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output


##############################################################################
# Classes
##############################################################################
class RecLoss(nn.Module):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, mse_loss=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if mse_loss else nn.BCELoss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, inputs, target_is_real):
        # if input is a list
        all_losses = []
        for input in inputs:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss_input = self.loss(input, target_tensor)
            all_losses.append(loss_input)
        # 这里是求和不是求mean
        loss = sum(all_losses)
        return loss, all_losses


#

#定义模式损失
class PatternLoss_2D(nn.Module):
    def __init__(self,MSE_Loss=True,dilation=1):
        super(PatternLoss_2D, self).__init__()
        if MSE_Loss:
            self.loss = nn.MSELoss()  #MSE距离
        else:
            self.loss=nn.L1Loss() #L1 loss
            #nn.SmoothL1Loss
        # 保存卷积核的dilation.
        self.dilation=dilation

# 自定义卷积核的权重
    def conv_init(self,conv):
        weights = torch.Tensor([[256, 128, 64], [32, 16, 8], [4, 2, 1]])
        # 这里只需要定义batch size的数量为1即可，做conv2D的时候，会按照batch size分开做，然后拼接起来
        #
        weights = weights.expand(1, 1, 3, 3)
        # print(weights)
        conv.weight.data = weights.cuda()

    def Convert_to_0_255(self, input):

        return input.mul(0.5).add(0.5).mul(255).cuda()

    def Get_3X3_Hist(self,input):

        input=input.cuda()
        input_processed=input>GRAY_THRESHOLD  # 阈值分割。input_processed为0或者1,type为torch.uint8
        # eg:>> a=torch.Tensor([2,4,6])
        #    >>print(a>3)
        #    tensor([0, 1, 1], dtype=torch.uint8)

        input_processed=input_processed.float()#type由torch.uint8转为torch.FloatTensor

        #print("input after processing:", input_processed)

        MyTemplate_Conv=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=0,dilation=self.dilation,bias=False)

        MyTemplate_Conv.apply(self.conv_init)

        # input_processed的shape为[B,C,H,W],卷积核的shape为[1,1,3,3],则Hist的输出shape为[B,1,H-2,W-2]
        Hist=MyTemplate_Conv(input_processed)  #计算统计直方图[B,1,H-2,W-2]

        Hist=Hist.reshape(-1)  # 拉成一维
        Hist=Hist.int()
        Count=torch.bincount(Hist.cpu(),minlength=512)#计数

        #-------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,没有小数------------#
        Count = Count.float()
        # -------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,没有小数------------#
        Count=Count/len(Hist)
        return Count.cuda()

    #input和target的格式[batch,C,H,W]
    def  PatternLoss_for_2D(self,input, target):

        if (input.size() != target.size()):
            raise Exception("the shapes of input and target must be the same!")

        N,C,H,W=input.size()  #C为单通道
        #[-1，-1]转为[0,255]
        image_input = self.Convert_to_0_255(input)
        image_target =self.Convert_to_0_255(target)

        #得到直方图
        Input_Hist=self.Get_3X3_Hist(image_input) # 3X3的模板遍历图像，并转成10进制后的直方图，范围为[0,512]

        Target_Hist =self.Get_3X3_Hist(image_target)

        s=self.loss(Input_Hist,Target_Hist)

        s = s / N / C
        return Variable(s,requires_grad=True)

    def __call__(self, input, target):

        return  self.PatternLoss_for_2D(input.cuda(),target.cuda())
        #return  self.PatternLoss_for_2D(input,target)


# 定义模式损失
class PatternLoss_3D(nn.Module):
    def __init__(self, MSE_Loss=True, dilation=1):
        super(PatternLoss_3D, self).__init__()
        if MSE_Loss:
            self.loss = nn.MSELoss()  # MSE距离
        else:
            self.loss = nn.L1Loss()  # L1 loss
            # nn.SmoothL1Loss
        # 保存卷积核的dilation.
        self.dilation = dilation

    # 自定义卷积核的权重
    def conv_init(self, conv):
        weights = torch.Tensor([[256, 128, 64], [32, 16, 8], [4, 2, 1]])
        # 扩展为三维结构的通道数
        # B, C, H, W = conv.weight.data.shape
        weights = weights.expand(1, 1, 3, 3)
        # print(weights)
        conv.weight.data = weights.cuda()

    def Segment_and_Convert_to_0_to_1(self, input):

        # output = input.mul(0.5).add(0.5).mul(255).cuda()
        t=(GRAY_THRESHOLD/255-0.5)/0.5
        output = input >=t   # 阈值分割。input_processed为0或者1,type为torch.uint8
        return output.float()  # type由torch.uint8转为torch.FloatTensor

    def Get_3X3_Hist_3D(self, input):

        input = input.cuda()

        # 按照dim=1进行切分，即[B,C,H,W]-->C*[B,1,H,W],返回值是一个tuple
        MyTemplate_Conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0,
                                    dilation=self.dilation, bias=False)
        MyTemplate_Conv.apply(self.conv_init)

        splite_interval = 1
        splited_imgs = torch.split(input, splite_interval, dim=1)

        for i in range(len(splited_imgs)):
            if i == 0:
                total_Hist = MyTemplate_Conv(splited_imgs[i])  # 计算统计直方图[Batch,C,H-2,W-2]
            else:
                Hist = MyTemplate_Conv(splited_imgs[i])  # 计算统计直方图[Batch,C,H-2,W-2]
                total_Hist = torch.cat((total_Hist, Hist), dim=1)

        Hist = total_Hist.reshape(-1)  # 拉成一维
        Hist = Hist.int()
        Count = torch.bincount(Hist.cpu(), minlength=512)  # 计数
        # -------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,没有小数------------#
        Count = Count.float()
        # -------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,没有小数------------#
        Count = Count / len(Hist)
        return Count.cuda()

    # input和target的格式[batch,C,H,W]
    def PatternLoss_for_3D(self, input, target):

        if (input.size() != target.size()):
            raise Exception("the shapes of input and target must be the same!")

        N, C, H, W = input.size()
        # 得到直方图
        Input_Hist = self.Get_3X3_Hist_3D(input)  # 3X3的模板遍历图像，并转成10进制后的直方图，范围为[0,512]
        Target_Hist = self.Get_3X3_Hist_3D(target)
        s = self.loss(Input_Hist, Target_Hist)
        s = s / N / C
        return Variable(s, requires_grad=True)

    def Directional_Pattern_loss(self, input, target, mode="xy"):

        if mode not in ["xy", "yz", "zx"]:
            raise ValueError("parameter is set wrongly")
        if (input.size() != target.size()):
            raise Exception("the shapes of input and target must be the same!")

        if mode == "yz":
            input = input.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
            target = target.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
        elif mode == "zx":
            input = input.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
            target = target.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]

        # image_input = self.Segment_and_Convert_to_0_to_1(input)
        # image_target = self.Segment_and_Convert_to_0_to_1(target)

        # 为了快速计算，转换了B和C的维度，这里是等效的。
        input = input.permute(1, 0, 2, 3)  # 图像维度变化：[B,C,H,W]-->[C,B,H,W]
        target = target.permute(1, 0, 2, 3)  # 图像维度变化：[B,C,H,W]-->[C,B,H,W]

        loss = self.PatternLoss_for_3D(input, target)
        return loss

    def Total_Pattern_loss(self, input, target):

        # [-1，-1]转到[0,255]，并做阈值分割转为[0.0,1.0]
        image_input = self.Segment_and_Convert_to_0_to_1(input)
        image_target = self.Segment_and_Convert_to_0_to_1(target)

        # xy方向
        loss_xy = self.Directional_Pattern_loss(image_input, image_target,"xy")

        # yz方向
        # image_input_yz = image_input.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
        # image_target_yz = image_target.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
        loss_yz = self.Directional_Pattern_loss(image_input, image_target,"yz")

        # zx方向
        # image_input_zx = image_input.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
        # image_target_zx = image_target.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
        loss_zx = self.Directional_Pattern_loss(image_input, image_target,"zx")

        return (loss_xy + loss_yz + loss_zx)/3

    def __call__(self, input, target):

        return self.Total_Pattern_loss(input.cuda(), target.cuda())


# ##定义孔隙度损失
# class PorosityLoss(nn.Module):
#     def __init__(self, MSE_Loss=True):
#         super(PorosityLoss, self).__init__()
#         if MSE_Loss:
#             self.loss = nn.MSELoss()  # 欧式距离
#         else:
#             self.loss = nn.L1Loss()  # L1 loss
#
#     def Segment_and_Convert_to_0_to_1(self, input):
#
#         # return input.mul(0.5).add(0.5).mul(255).cuda()
#         t = (GRAY_THRESHOLD / 255 - 0.5) / 0.5
#         output = input >= t  # 阈值分割。input_processed为0或者1,type为torch.uint8
#         return output.float()  # type由torch.uint8转为torch.FloatTensor
#
#     def Get_Porosity(self, input):
#
#         input = input.cuda()
#         Hist = input.reshape(-1)  # 拉成一维
#         Hist = Hist.int()
#         Count = torch.bincount(Hist.cpu(), minlength=2)  # 计数
#
#         # print('Count is:',Count)
#
#         # -------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法,可能得到的值全为0------------#
#         Count = Count.float()
#         # -------注意这里必须要先转成FloatTensor，不然Count/len(Hist)就是整数除法，可能得到的值全为0------------#
#         Count = Count / len(Hist)
#         # Count是大小为2的向量，黑点对应着index 0,白点对应index 1，Count.data[1]即为孔隙度.
#         return Count.data[1].cuda()
#
#     # input和target的格式[batch,C,H,W]
#     def Cal_PorosityLoss(self, input, target):
#
#         if (input.size() != target.size()):
#             raise Exception("the shapes of input and target must be the same!")
#
#         N, C, H, W = input.size()  # C为单通道
#         # 分割
#         image_input = self.Segment_and_Convert_to_0_to_1(input)
#         image_target = self.Segment_and_Convert_to_0_to_1(target)
#
#         # 得到直方图
#         Input_Hist = self.Get_Porosity(image_input)
#         Target_Hist = self.Get_Porosity(image_target)
#         s = self.loss(Input_Hist, Target_Hist)
#         s = s / N / C
#         return Variable(s, requires_grad=True)
#
#     def __call__(self, input, target):
#
#         return self.Cal_PorosityLoss(input.cuda(), target.cuda())



# 定义孔隙度loss

##定义孔隙度损失，可用于2D或者3D

class PorosityLoss_3D(nn.Module):
    def __init__(self, MSE_Loss=True):
        super(PorosityLoss_3D, self).__init__()
        if MSE_Loss:
            self.loss = nn.MSELoss()  # 欧式距离
        else:
            self.loss = nn.L1Loss()  # L1 loss

    def Get_Porosity(self, input):

        input_flaten = input.flatten()
        # Thre = 0.0
        Thre = (GRAY_THRESHOLD / 255 - 0.5) / 0.5
        # eps = 1e-8
        eps = 1e-6
        input_seg = torch.nn.ReLU()(input_flaten - Thre) / (input_flaten - Thre + eps)
        porosity = input_seg.sum() / torch.numel(input_seg) #除以input_seg所有的元素，得到孔隙度
        return porosity

    def Cal_PorosityLoss(self, input, target):

        if (input.size() != target.size()):
            raise Exception("the shapes of input and target must be the same!")
        input_porosity = self.Get_Porosity(input)
        target_porosity = self.Get_Porosity(target)
        s = self.loss(input_porosity, target_porosity)
        return Variable(s, requires_grad=True)

    def __call__(self, input, target):

        return self.Cal_PorosityLoss(input, target)

# 硬数据L1 loss

class Harddata_L1_Loss(nn.Module):
    def __init__(self, L1_Loss=True):
        super(Harddata_L1_Loss, self).__init__()
        if L1_Loss:
            self.loss = nn.L1Loss()  # L1 loss
        else:
            self.loss = nn.MSELoss()  # 欧式距离

    # input和target的格式[batch,C,H,W]
    def Cal_HarddataLoss(self, input, target):
        if (input[0][0].size() != target[0][0].size()):
            raise Exception("the shapes of input and target must be the same!")
        N, C, H, W = input.size()  # C为单通道
        s = self.loss(input[:, 0, :, :], target[:, 0, :, :])
        loss = s / N
        return loss

    def __call__(self, input, target):

        return self.Cal_HarddataLoss(input, target)

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic',add_conv=False,noise_expand=True):
        super(G_Unet_add_input, self).__init__()
        self.nz = nz
        self.noise_expand=noise_expand
        max_nchn = 8
        # construct unet structure

        # 这里就是常规的UNet,只不过代码的实现看起来不那么直观。
        # 这里网络的定义顺序是：先定义最中间那层，然后作为submodule往两边扩展;再利用cat函数进行skip connection；继续更新submodule
        # 注意这里的标志位innermost，代表着第一层
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        ## 64-->128-->64
        # unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
        #                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        ## 128-->128-->128
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        # 注意这里的标志位outermost，代表着最后一层

        ## 1+nz-->64-->128

        # unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,add_conv=add_conv)

        ## 1+nz-->128-->128
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf * 2, unet_block,outermost=True,
        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,add_conv=add_conv)

        self.model = unet_block

    def forward(self, x, z=None):
        # 默认nz=8,噪声的通道数是8
        if self.nz > 0:
            ##扩展
            if self.noise_expand:
                z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                    z.size(0), z.size(1), x.size(2), x.size(3))
                x_with_z = torch.cat([x, z_img], 1)

            ###################直接生成BxnzxHxW############################
            else:
                z_img = None
                noise_shape=(x.shape[0],self.nz,x.shape[2],x.shape[3])
                if not torch.cuda.is_available():
                    z_img = torch.randn(noise_shape)   # 测试噪声直接生成BxnzxHxW的
                else:
                    z_img = torch.randn(noise_shape).cuda()
                x_with_z = torch.cat([x, z_img], 1)
        ###################直接生成BxnzxHxW############################

        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero',add_conv=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        # 每次降采样为原来size的一半
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        # 归一化
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        # 激活函数
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            # 上采样为2倍的尺寸
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            # 下采样为原来的一半
            down = downconv
            # up = [uprelu] + upconv + [nn.Tanh()] 原来的
            # revised by fjx-20190612
            if not add_conv:
                up = [uprelu] + upconv + [nn.Tanh()]  # 原来的
            else:
                #conv_1x1 = Conv2D_1x1(input_nc=outer_nc, output_nc=outer_nc, n_layers=1, norm_layer=nn.BatchNorm2d, nl_layer=nl_layer)
                conv_1x1 = Conv2D_1x1(input_nc=outer_nc, output_nc=outer_nc, n_layers=1, norm_layer=norm_layer, nl_layer=nl_layer)
                up = [uprelu] + upconv + [conv_1x1] + [nn.Tanh()]
            #
            model = down + [submodule] + up
        elif innermost:

            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:

            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # 输入和输出拼起来
            # skip connection
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        # 由于AvgPool2的存在，conv(x)出来尺寸缩小为原来的1/2,shortcut(x)也是原来的1/2
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            # 每一个BasicBlock出来的尺寸为原来的1/2
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        # nn.AvgPool2d(8)出来的尺寸为原来的1/8
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        #综上，conv出来的尺寸为原来的1/(16*8),即由128*128变成了1*1
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        # 拉成一个一维向量
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        # 外面指定了vaeLike为True
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
            # VAE输出两个向量，一个是均值，另一个是方差
            # output: latent mean,outputVar: latent log variance
        else:
            return output
        return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic',add_conv=False):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        # 增加了add_conv的标志位，设置为False即为原始功能，设置为True为在最后一层添加1x1 Conv
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,add_conv=add_conv)


        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero',add_conv=False):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        # 噪声与输入cat到一起
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)# 原来的

            # # 先放大，再拉到指定通道
            # upconv = upsampleLayer(
            #     inner_nc * 2, inner_nc * 2, upsample=upsample, padding_type=padding_type)
            # # 先放大，再拉到指定通道
            # conv_1x1 = Conv2D_1x1(input_nc=inner_nc * 2, output_nc=outer_nc, n_layers=1, norm_layer=nn.BatchNorm2d,
            #                       nl_layer=nl_layer)
            down = downconv
            # revised by fjx-20190510
            if not add_conv:
                up = [uprelu] + upconv + [nn.Tanh()]  # 原来的
                # up = [uprelu] + upconv + [conv_1x1] + [nn.Tanh()]  # 改进1
            else:
                # conv_1x1_fuse = Conv3D_1x1(input_nc=1, output_nc=1, n_layers=1, norm_layer=nn.BatchNorm3d, nl_layer=nl_layer)
                conv_1x1_fuse = Conv2D_1x1(input_nc=outer_nc, output_nc=outer_nc, n_layers=1, norm_layer=nn.InstanceNorm2d, nl_layer=nl_layer)
                # up = [uprelu] + upconv + [conv_1x1]+ [conv_1x1_fuse] + [nn.Tanh()] # 改进2
                up = [uprelu] + upconv + [conv_1x1_fuse] + [nn.Tanh()] # 改进2

        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:

            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)

        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


class Conv3D_1x1(nn.Module):
    def __init__(self, input_nc=128, output_nc=1, n_layers=1, norm_layer=None, nl_layer=None):
        super(Conv3D_1x1, self).__init__()
        self.basic_sequence=[]
        # 先做norm
        if norm_layer is not None:
            self.basic_sequence += [norm_layer(input_nc)]
        # print("-----norm_layer done ------")
        self.basic_sequence += [nl_layer(), nn.Conv3d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)]

        self.sequence = []
        for n in range(0, n_layers):
            self.sequence += self.basic_sequence

        # self.sequence += [torch.nn.Tanh()]
        self.conv = nn.Sequential(*self.sequence)

    def forward(self, x):

        # return self.conv(x)

        # 判断x应为5维数据：(B,C,L,H,W),先升后降
        if x.dim() != 5:
            # 增加一维通道
            x = x.unsqueeze(dim=1)
        x = self.conv(x)
        # 增加一维通道
        x = x.squeeze(dim=1)
        return x

class Conv2D_1x1(nn.Module):
    def __init__(self, input_nc=128, output_nc=128, n_layers=1, norm_layer=None, nl_layer=None):
        super(Conv2D_1x1, self).__init__()
        self.basic_sequence=[]
        # 先做norm
        if norm_layer is not None:
            self.basic_sequence += [norm_layer(input_nc)]
        # print("-----norm_layer done ------")
        self.basic_sequence += [nl_layer(), nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)]

        self.sequence = []
        for n in range(0, n_layers):
            self.sequence += self.basic_sequence

        # self.sequence += [torch.nn.Tanh()]
        self.conv = nn.Sequential(*self.sequence)

    def forward(self, x):
        return self.conv(x)

class Conv2D_3x3(nn.Module):
    def __init__(self, input_nc=128, output_nc=128, n_layers=1, norm_layer=None, nl_layer=None):
        super(Conv2D_3x3, self).__init__()
        self.basic_sequence=[]
        # 先做norm
        if norm_layer is not None:
            self.basic_sequence += [norm_layer(input_nc)]
        # print("-----norm_layer done ------")
        self.basic_sequence += [nl_layer(), nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]

        self.sequence = []
        for n in range(0, n_layers):
            self.sequence += self.basic_sequence

        # self.sequence += [torch.nn.Tanh()]
        self.conv = nn.Sequential(*self.sequence)

    def forward(self, x):
        return self.conv(x)

# m = Conv_1x1(10, 10, n_layers=3,norm_layer=nn.InstanceNorm2d, nl_layer=nn.ReLU)
# print(m)
# x = torch.rand(4, 10, 10, 10)
# print(m(x).shape)
# print(m(x))



# m=Conv_1x1(128,128,norm_layer=nn.InstanceNorm2d,nl_layer=nn.ReLU)
# x=torch.rand(4,128,128,128)
# print(m(x).shape)