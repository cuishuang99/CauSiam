"""Print parameter count and FLOPs for the deblur model."""

from utils.img_utils import *
from utils.metric_cal import *
import argparse
from model.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='val')
parser.add_argument('--num_res', type=int, default=2)
parser.add_argument('--nc', type=int, default=16)
parser.add_argument('--pixel_weight', type=float, default=1.0)
parser.add_argument('--fft_weight', type=float, default=0.1)
args = parser.parse_args()
model = Model(args)

print('Parameters: {:.3f} M'.format(model.cal_parameters() / 1e6))
print('FLOPs: {:.4f} G'.format(model.cal_floaps() / 1e9))
