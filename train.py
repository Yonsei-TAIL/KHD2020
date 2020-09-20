from __future__ import print_function

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import random
import numpy as np

from options import parse_option
from network import create_model
from utils import get_optimizer, get_loss_function, lr_update
from utils.core import train, validate
from datasets import get_dataloader
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
    opt = parse_option(print_option=True)

    # Data Loader
    dataset_trn, dataset_val = get_dataloader(opt)

    # Network
    net = create_model(opt)

    # Loss Function
    criterion = get_loss_function(opt)

    # Optimizer
    optimizer = get_optimizer(net, opt)

    # Initial Best Score
    best_acc, best_epoch = [0, 0]

    for epoch in range(opt.start_epoch, opt.max_epoch):
        # Train
        train(net, dataset_trn, optimizer, criterion, epoch, opt)

        # Evaluate
        best_acc, best_epoch = validate(dataset_val, net, criterion, optimizer, epoch, opt, best_acc, best_epoch)

        lr_update(epoch, opt, optimizer)

    print('Training done')