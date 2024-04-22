import argparse
import os
from util import util
import torch
import models
import data

from options.parameters import *


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=False, default=DATAROOT,help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='input batch size')
        # parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        # parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--loadSize', type=int, default=REC_SIZE, help='scale images to this size')  # revised by fjx 20181213
        parser.add_argument('--fineSize', type=int, default=REC_SIZE, help='then crop to this size')
        # parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')  # revised by fjx 20181213
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        # parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')  # revised by fjx 20181213
        parser.add_argument('--output_nc', type=int, default=REC_SIZE, help='# of output image channels')
        # 噪声的通道数
        parser.add_argument('--nz', type=int, default=NOISE_SIZE, help='#latent vector')
        parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        # parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in first conv layer')
        # parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        # parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        # 保存文件的路径
        parser.add_argument('--name', type=str, default=NAME, help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='not implemented')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='aligned,single')
        parser.add_argument('--model', type=str, default=MODEL, help='chooses which model to use. bicycle,, ...')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model') # commented by fjx 20181222
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        # parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator') # commented by fjx 20181221
        parser.add_argument('--use_dropout', type=bool, default=True,help='use dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # 实际训练时，不需要no_flip
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        # models
        parser.add_argument('--num_Ds', type=int, default=2, help='number of Discrminators')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='dcgan|lsgan')

        # 128重建
        # parser.add_argument('--netD', type=str, default='basic_128_multi', help='selects model to use for netD')
        # parser.add_argument('--netD2', type=str, default='basic_128_multi', help='selects model to use for netD')
        # parser.add_argument('--netG', type=str, default='unet_128', help='selects model to use for netG')
        # parser.add_argument('--netE', type=str, default='resnet_128', help='selects model to use for netE')

        # 256重建
        parser.add_argument('--netD', type=str, default='basic_%d_multi' % REC_SIZE,help='selects model to use for netD')  # revised by fjx 20181213
        parser.add_argument('--netD2', type=str, default='basic_%d_multi' % REC_SIZE,help='selects model to use for netD')  # revised by fjx 20181213
        parser.add_argument('--netG', type=str, default='unet_%d' % REC_SIZE,help='selects model to use for netG')  # revised by fjx 20181213
        parser.add_argument('--netE', type=str, default='resnet_%d' % REC_SIZE,help='selects model to use for netE')  # revised by fjx 20181213

        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
        parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')

        # extra parameters
        parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        # parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        parser.add_argument('--conditional_D', action='store_true', help='if use conditional GAN for D')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        # parser.add_argument('--center_crop', action='store_true', help='if apply for center cropping for the test') commented by fjx 20181221
        parser.add_argument('--center_crop', type=bool,default=False, help='if apply for center cropping for the test')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--add_conv', type=bool, default=ADD_CONV,help='if add extra 1x1 Conv2D after the last layer or not')
        parser.add_argument('--three_directions', type=bool, default=True,help='if use three directions (x,y,z) to  discriminate fake and real by discriminator')
        parser.add_argument('--noise_expand', type=bool, default=True,help='True for noise expand (nzx1x1-->nzxHxW),False for no expand (initial dimension is nzxHxW)')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)

        if self.isTrain:  # train or test:
            file_name = os.path.join(expr_dir, 'opt_train.txt')
        else:
            file_name = os.path.join(expr_dir, 'opt_test.txt')

        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
