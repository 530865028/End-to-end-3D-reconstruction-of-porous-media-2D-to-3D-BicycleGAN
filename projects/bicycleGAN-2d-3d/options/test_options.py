from .base_options import BaseOptions
from options.parameters import *
import torch

if torch.cuda.is_available():
    CUDA='0'
else:
    CUDA = '-1'

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default=TEST_CHECKPOINT_DIR, help='models are saved here')

        parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help='saves results here.')
        # parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc') # commented by fjx
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # 读入num_test幅图像进行测试
        parser.add_argument('--num_test', type=int, default=2, help='how many test images to run')
        # 每一幅图像生成n_samples个样本
        parser.add_argument('--n_samples', type=int, default=1, help='#samples')
        # parser.add_argument('--no_encode', action='store_true', help='do not produce encoded image')
        parser.add_argument('--no_encode', type=bool, default=True, help='do not produce encoded image')
        parser.add_argument('--sync', action='store_true', help='use the same latent code for different input images')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        # 是否保留图像的顺序
        parser.add_argument('--serial_batches', type=bool, default=True,help='if true, takes images in order to make batches, otherwise takes them randomly') # commented by fjx 20181213
        # gpu_ids='-1'，即为CPU模式
        parser.add_argument('--gpu_ids', type=str, default= CUDA, help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        # 测试模式（非训练模式）
        self.isTrain = False
        return parser
