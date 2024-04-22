import torch
from .base_options import BaseOptions

from options.parameters import *

# if torch.cuda.is_available():
#     CUDA='0'
# else:
#     CUDA = '-1'
#     print("---------------NO GPUS FOUND!---------------")
#     print("---------------Program will be run on CPU!---------------")

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default=CHECKPOINTS_DIR, help='models are saved here')
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        # parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_ncols', type=int, default=10, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        # parser.add_argument('--display_winsize', type=int, default=256, help='display window size') revised by fjx 20181213
        parser.add_argument('--display_winsize', type=int, default=REC_SIZE, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        # parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        # 继续训练的标志
        parser.add_argument('--continue_train', type=bool, default=False,help='continue training: load the latest model')
        # 加载对应的模型，以epoch编号命名
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        #图像修复做了参数修改
        # parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        # 图像修复做了参数修改
        # parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # learning rate
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        #学习率的调整
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
        # lambda parameters
        # 可通过指定lambda是否为0，来决定是否要使用对应的网络。如单独测试CVAE-GAN，以做对比。
        # parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')
        parser.add_argument('--lambda_L1', type=float, default=lamb_L1, help='weight for |B-G(A, E(B))|')
        # 三维结构G(A, random_z)的第一层G(A, random_z)[0]与输入图像的差异
        parser.add_argument('--lambda_L1_Harddata', type=float, default=lamb_L1_Harddata, help='weight for |G(A, random_z)[0]-A|')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')
        parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')
        parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')

        # remain_area=10,20的参数
        parser.add_argument('--lambda_pattern', type=float, default=lamb_Pattern, help='weight for Pattern loss')
        parser.add_argument('--lambda_pattern_multiscale', type=float, default=lamb_Pattern_multiscale, help='weight for Pattern loss')
        parser.add_argument('--lambda_porosity', type=float, default=lamb_Porosity, help='weight for Porosity loss')
        # 混合的时候(remain_area=1,3,5,10)的参数
        # parser.add_argument('--lambda_pattern', type=float, default=5*1e5, help='weight for Pattern loss')
        # parser.add_argument('--lambda_porosity', type=float, default=1e4, help='weight for Porosity loss')

        parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')
        parser.add_argument('--serial_batches', type=bool, default=False,help='if true, takes images in order to make batches, otherwise takes them randomly') # commented by fjx 20181213
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')


        # 训练模式
        self.isTrain = True
        return parser
