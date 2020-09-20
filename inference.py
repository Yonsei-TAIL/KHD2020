from __future__ import print_function

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

import os
import random
import numpy as np
from glob import glob

from options import parse_option
from network import create_model

import warnings
warnings.filterwarnings('ignore')


# Seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

#NOTE: main loop for training
if __name__ == "__main__":
    # Option
    opt = parse_option(print_option=False)

    # Network
    net = create_model(opt)

    # Load Data List
    imgList = glob(os.path.join(opt.data_root, '*'))

    # Inference
    for img_path in imgList:
        raise NotImplementedError