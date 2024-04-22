import random
import torch
from .base_model import BaseModel
from . import networks
import copy
import torch.nn as nn
class BiCycleGANModel(BaseModel):
    def name(self):
        return 'BiCycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        if opt.isTrain:
            # assert opt.batch_size % 2 == 0  # load two images at one time.
            # revised by fjx 20190604
            pass

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl','L1_Harddata','G_PatternLoss','G_PatternLossMultiScale','G_Porosity']
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl','G_PatternLoss','G_PatternLossMultiScale','G_Porosity']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # 定义需要可视化的图像，直接将名字拷贝过来即可。
        self.visual_names = ['real_A_encoded',
                             'real_B_encoded_xy_slice',
                             'real_B_encoded_yz_slice',
                             'real_B_encoded_zx_slice',
                             'fake_B_random_xy_slice',
                             'fake_B_random_yz_slice',
                             'fake_B_random_zx_slice',
                             'fake_B_encoded_xy_slice',
                             'fake_B_encoded_yz_slice',
                             'fake_B_encoded_zx_slice'
                             ]

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        # opt.isTrain 在train的时候为True，在test的时候为False
        # no_encode默认为不启用.
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']

        # 根据base_option、train_option以及对应的train脚本，可以得出以下的参数配置：
        # 输入通道opt.input_nc=1(原来为3)
        # 输出通道opt.output_nc=1(原来为3)
        # 噪声维度opt.nz=8
        # G基本的通道数opt.ngf=64
        # G网络opt.netG='unet_128'，UNet with skip connection
        # 归一化方法opt.norm=instance normalization,而不是BN.
        # 激活函数opt.nl='relu'
        # 是否使用use_dropout，默认使用
        # 网络初始化参数类型opt.init_type，默认为xavier，是一种均匀分布
        # gpu_ids，GPU的ID
        # where_add: help='input|all|middle; where to add z in the network G'
        # opt.upsample: default='basic', help='basic | bilinear'. basic意思为调用Deconv函数; bilinear是调用ReflectionPad2d函数
        #

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample,add_conv=opt.add_conv,noise_expand=opt.noise_expand)

        # 鉴别器的输出通道数。opt.conditional_D默认不启用。
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        # opt.gan_mode: 默认值为'lsgan', help='dcgan|lsgan'
        use_sigmoid = opt.gan_mode == 'dcgan'
        # opt.num_Ds默认两个，即两个鉴别器
        if use_D:
            # 第一个D 为'basic_128_multi'
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            # 第二个D也为'basic_128_multi'
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_E:
            self.model_names += ['E']
            # E的默认构成是'resnet_128'
            # use_vae默认为True
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()

            # added by fjx 20181220
            # self.criterionPattern = networks.PatternLoss_2D(MSE_Loss=True, dilation=1)
            self.criterionPattern = networks.PatternLoss_3D(MSE_Loss=True, dilation=1)

            # self.criterionPatternMultiScale = networks.PatternLoss_2D(MSE_Loss=True, dilation=2)
            self.criterionPatternMultiScale = networks.PatternLoss_3D(MSE_Loss=True, dilation=2)

            self.criterionPorosity = networks.PorosityLoss_3D(MSE_Loss=True)
            self.criterion_Harddata = networks.Harddata_L1_Loss(L1_Loss=True)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        # AtoB是一个标志位
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # added by fjx -20181225
    def set_single_input(self, input):
        self.real_A = input

    # 默认为高斯分布
    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        # 这里的 mu, logvar的含义是指VAE编码输出的两个向量，一个是均值mu,一个是Log方差logvar
        # mu:latent mean, logvar: latent log variance.

        mu, logvar = self.netE.forward(input_image)
        # 对logvar进行逐元素操作 exp(0.5*logvar[i])

        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        # z
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    # 测试，输入是real_A和z
    # 可选择输入real_B来进行编码

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.fake_B
            # return self.real_A, self.fake_B, self.real_B   # commented by fjx 20190424

    # added by fjx 20181225
    def test_for_output(self, z0=None):
        with torch.no_grad():
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.fake_B

    def get_input_and_target(self):

        return self.real_A, self.real_B

    # 前向传播，只有G和E, 可以看出G和E是两个网络共享的

    def forward(self):

        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        # 切割为两等份, 作者表示为了增加多样性？
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]


        # added by fjx 20190522
        # get real images
        # half_size = self.opt.batch_size // 2
        # self.real_A_encoded = self.real_A
        # self.real_B_encoded = self.real_B
        # self.real_B_random = self.real_B

        # get encoded z
        '''# 这里表示的是cVAE-GAN中的E'''

        # 利用visdom对随机取的三个方向的切片可视化
        self.real_B_encoded_xy_slice = self.get_a_rand_slice(self.real_B_encoded, mode="xy")
        self.real_B_encoded_yz_slice = self.get_a_rand_slice(self.real_B_encoded, mode="yz")
        self.real_B_encoded_zx_slice = self.get_a_rand_slice(self.real_B_encoded, mode="zx")

        # print('\n-------shape of self.fake_B_random in bicycle_gan_model.forward:---------', self.real_B_encoded.shape)
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)


        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        '''# 这里表示的是cVAE-GAN中的G'''
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)

        # 利用visdom对随机取的三个方向的切片可视化
        self.fake_B_encoded_xy_slice = self.get_a_rand_slice(self.fake_B_encoded, mode="xy")
        self.fake_B_encoded_yz_slice = self.get_a_rand_slice(self.fake_B_encoded, mode="yz")
        self.fake_B_encoded_zx_slice = self.get_a_rand_slice(self.fake_B_encoded, mode="zx")

        # generate fake_B_random
        # real_A +rand z 输入到G中
        '''# 这里表示的是cLR-GAN中的G'''
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)
        # print("self.fake_B_random.shape:",self.fake_B_random.shape)

        # 将硬数据拷贝到对应的位置
        # [::,1::,...]这里切片是代表着不要第一层
        self.fake_B_random = torch.cat((self.real_A_encoded,self.fake_B_random[::,1::,...]),dim=1)

        # 利用visdom对随机取的三个方向的切片可视化

        self.fake_B_random_xy_slice=self.get_a_rand_slice(self.fake_B_random,mode="xy")
        self.fake_B_random_yz_slice=self.get_a_rand_slice(self.fake_B_random,mode="yz")
        self.fake_B_random_zx_slice=self.get_a_rand_slice(self.fake_B_random,mode="zx")

        # 默认conditional_D不启动，作者表示为了增加样本的多样性。
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        else:
            # 编码器的输出z_encoded和real_A[:half_size]作为G的输入，输出self.fake_data_encoded
            self.fake_data_encoded = self.fake_B_encoded
            # real_A[:half_size]和rand noise作为G的输入，输出 self.fake_data_random
            self.fake_data_random = self.fake_B_random
            # 编码器的输入real_B[0:half_size]
            self.real_data_encoded = self.real_B_encoded
            # 另一半real_B，real_B[half_size:]
            self.real_data_random = self.real_B_random

        # compute z_predict
        # lambda_z的默认值为0.5
        '''# 这里是cLR-GAN中的E'''
        if self.opt.lambda_z > 0.0:
            # print('\n-------shape of self.fake_B_random in bicycle_gan_model.forward:---------', self.fake_B_random.shape)
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate


    def criterionGAN_three_directions(self, netD, input, Lable):

        pred_1 = netD(input.detach())
        loss_1, _ = self.criterionGAN(pred_1, Lable)

        input_2 = input.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
        pred_2 = netD(input_2.detach())
        loss_2, _ = self.criterionGAN(pred_2, Lable)

        input_3 = input.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
        pred_3 = netD(input_3.detach())
        loss_3, _ = self.criterionGAN(pred_3, Lable)

        loss_total = sum([loss_1, loss_2, loss_3]) / 3.0
        return loss_total


    # D依然是常规的二分类问题，默认是LSGAN
    def backward_D(self, netD, real, fake, three_directions = False):
        # Fake, stop backprop to the generator by detaching fake_B
        if not three_directions:
            pred_fake = netD(fake.detach())
            # real
            pred_real = netD(real)
            loss_D_fake, _ = self.criterionGAN(pred_fake, False)
            loss_D_real, _ = self.criterionGAN(pred_real, True)
        else:
            loss_D_fake = self.criterionGAN_three_directions(netD, fake, False)
            loss_D_real = self.criterionGAN_three_directions(netD, real, True)

        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    # GAN的Loss
    def backward_G_GAN(self, fake, netD=None, ll=0.0, three_directions=False):
        if ll > 0.0:
            if not three_directions:
                pred_fake = netD(fake)
                loss_G_GAN, _ = self.criterionGAN(pred_fake, True)

            else:
                # print('-----------fake.data.device-------------',fake.data.device)
                pred_fake_1 = netD(fake)
                loss_G_GAN_1, _ = self.criterionGAN(pred_fake_1, True)
                # print('----------pred_fake_1----------')

                fake_2 = fake.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
                pred_fake_2 = netD(fake_2)
                loss_G_GAN_2, _ = self.criterionGAN(pred_fake_2, True)
                # print('----------pred_fake_2----------')
                fake_3 = fake.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
                pred_fake_3 = netD(fake_3)
                loss_G_GAN_3, _ = self.criterionGAN(pred_fake_3, True)
                # print('----------pred_fake_3----------')
                loss_G_GAN = (loss_G_GAN_1 + loss_G_GAN_2 + loss_G_GAN_3) / 3.0
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN, three_directions=self.opt.three_directions)
        # opt.lambda_GAN和opt.lambda_GAN2 默认值都是1.0
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2, three_directions=self.opt.three_directions)
        else:
        # 默认是两个D
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2, three_directions=self.opt.three_directions)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            # 计算kl散度  loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        #
        # if self.opt.lambda_L1_Harddata > 0.0:
        #     self.loss_L1_Harddata=self.criterion_Harddata(self.fake_data_random,self.real_A_encoded)*self.opt.lambda_L1_Harddata

        # else:
        #     self.loss_L1_Harddata=0.0

            # added by fjx 20181220
        # Pattern Loss
        if self.opt.lambda_pattern > 0.0:
            # self.fake_data_random 是G(noise,real_A),real_B_encoded是target
            self.loss_G_PatternLoss=self.criterionPattern(self.fake_data_random,self.real_B_encoded)* self.opt.lambda_pattern
        else:
            self.loss_G_PatternLoss=0.0

        # MultiScale Pattern Loss
        if self.opt.lambda_pattern_multiscale > 0.0:
            # self.fake_data_random 是G(noise,real_A),real_B_encoded是target
            self.loss_G_PatternLossMultiScale=self.criterionPatternMultiScale(self.fake_data_random,self.real_B_encoded)* self.opt.lambda_pattern_multiscale
        else:
            self.loss_G_PatternLossMultiScale=0.0

        # Porosity Loss
        if self.opt.lambda_porosity > 0.0:
            # self.fake_data_random 是G(noise,real_A),real_B_encoded是target
            self.loss_G_Porosity=self.criterionPorosity(self.fake_data_random,self.real_B_encoded)* self.opt.lambda_porosity
        else:
            self.loss_G_Porosity=0.0


        # 叠加G的损失函数
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl \
                      + self.loss_G_Porosity \
                      + self.loss_G_PatternLoss \
                      + self.loss_G_PatternLossMultiScale \
                      # + self.loss_L1_Harddata
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded, three_directions= self.opt.three_directions)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random, three_directions= self.opt.three_directions)
            self.optimizer_D.step()
        # 更新D2
        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random, three_directions= self.opt.three_directions)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            #print(' self.loss_z_L1.shape', self.loss_z_L1.shape)
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        # 默认opt.lambda_z =0.5
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()

    def return_a_random_xy_slice(self, src_3D):
        img_3D = src_3D.detach()
        B, C, H, W = img_3D.shape
        r_b, r_l = random.randint(0, B - 1), random.randint(0, C - 1)
        rand_slice = img_3D[r_b, r_l, :, :]
        # 恢复成4维tensor
        rand_slice = rand_slice.unsqueeze(0)
        rand_slice = rand_slice.unsqueeze(0)
        return rand_slice

    def return_a_random_yz_slice(self, src_3D):
        img_3D = src_3D.detach()
        img_3D = img_3D.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
        B, C, H, W = img_3D.shape
        r_b, r_l = random.randint(0, B - 1), random.randint(0, C - 1)
        rand_slice = img_3D[r_b, r_l, :, :]
        # 恢复成4维tensor

        rand_slice = rand_slice.unsqueeze(0)
        rand_slice = rand_slice.unsqueeze(0)
        return rand_slice

    def return_a_random_zx_slice(self, src_3D):
        img_3D = src_3D.detach()
        img_3D = img_3D.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
        B, C, H, W = img_3D.shape
        r_b, r_l = random.randint(0, B - 1), random.randint(0, C - 1)
        rand_slice = img_3D[r_b, r_l, :, :]
        # 恢复成4维tensor
        rand_slice = rand_slice.unsqueeze(0)
        rand_slice = rand_slice.unsqueeze(0)
        return rand_slice

    # added by fjx 20190422，一个阳光明媚的日子。
    # mode should be in ["xy","yz","zx"]
    def get_a_rand_slice(self, src_3D, mode="xy"):

        if mode not in ["xy", "yz", "zx"]:
            raise ValueError("parameter mode is set wrongly!")
        img_3D = src_3D.detach()

        if mode == "yz":
            img_3D = img_3D.permute(0, 2, 1, 3)  # 图像维度变化：[B,C,H,W]-->[B,H,C,W]
        elif mode == "zx":
            img_3D = img_3D.permute(0, 3, 1, 2)  # 图像维度变化：[B,C,H,W]-->[B,W,C,H]
        # img_3D = img_3D.contiguous()
        B, C, H, W = img_3D.shape
        # r_b, r_l = random.randint(0, B - 1), random.randint(0, C - 1)
        r_l = random.randint(0, C - 1)
        r_b=0
        rand_slice = img_3D[r_b, r_l, :, :]
        # 恢复成4维tensor
        # 用rand_slice.unsqueeze_(0) in-place操作会报错
        rand_slice=rand_slice.unsqueeze(0)
        rand_slice=rand_slice.unsqueeze(0)

        # rand_slice=rand_slice.float()
        return rand_slice

