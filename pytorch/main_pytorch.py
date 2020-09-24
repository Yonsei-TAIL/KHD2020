import cv2
import random
import os, sys
import argparse
import numpy as np

from utils.data_loader import load_dataloader
from utils.model import load_model, bind_model, train_model, valid_model
from utils.optim_utils import lr_update, load_optimizer

import torch
import torch.nn as nn

import nsml

# Seed
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def ParserArguments():
    args = argparse.ArgumentParser()

    # Setting Hyperparameters
    args.add_argument('--nb_epoch', type=int, default=170)       # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=16)      # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)      # 분류될 클래스 수는 4개
    args.add_argument('--stack_channels', action='store_true')   # 2가지 window로 만들어낸 이미지를 input channel로 쌓아 3-channel로
    
    # Pre-processing
    args.add_argument('--img_size', type=int, default=224) # input crop image size
    args.add_argument('--w_min', type=int, default=50) # Window min
    args.add_argument('--w_max', type=int, default=180) # Window max
    args.add_argument('--zscore', action='store_true', help="apply ImageNet-based Z-score normalization")
    args.add_argument('--balancing_method', type=str, default='weights', help="'aug' : augmentation / 'weights' : class_weights")

    # Optimization Settings
    args.add_argument('--learning_rate', type=float, default=1e-3)  # learning rate 설정
    args.add_argument('--lr_decay_epoch', type=str, default='80,120,160')  # learning rate decay epoch
    args.add_argument('--optim', type=str, default='adam')  # Optimizer
    args.add_argument('--momentum', type=float, default=0.9)  # Momentum
    args.add_argument('--wd', type=float, default=3e-2)  # Weight decay
    args.add_argument('--bias_decay', action='store_true')  # 선언 시 bias에도 weight decay 적용
    
    # Network
    args.add_argument('--network', type=str, default='resnet34')
    args.add_argument('--resume', type=str, default='./weights/resnet34.pth')
    args.add_argument('--dropout', type=float, default=0.5)

    # Augmentation
    args.add_argument('--augmentation', type=str, default='light', help="'light' or 'heavy")          
    args.add_argument('--rot_factor', type=float, default=15)          
    args.add_argument('--scale_factor', type=float, default=0.15)          

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = args.parse_args()
    args.lr_decay_epoch = list(map(int, args.lr_decay_epoch.split(',')))

    return args

if __name__ == '__main__':
    args = ParserArguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model = load_model(args)
    model.to(device)
    bind_model(model, args)

    # Loss
    if args.balancing_method == 'weights':
        class_weights = torch.Tensor([1,4,6,9])
        criterion = nn.CrossEntropyLoss(class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = load_optimizer(model, args)


    if args.pause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())

    if args.mode == 'train':  ## for train mode
        print('Training start ...')
        batch_train, batch_val = load_dataloader(args)
        
        #####   Training loop   #####
        for epoch in range(args.nb_epoch):
            train_loss, train_f1 = train_model(epoch, batch_train, device, optimizer, model, criterion, args)
            val_loss, val_f1 = valid_model(epoch, batch_val, device, model, criterion, args)

            # total summary
            print("  * Train loss = {:.4f} | Train F1 = {:.4f} | Val loss = {:.4f} | Val F1 = {:.4f}"\
                    .format(train_loss.avg, train_f1, val_loss.avg, val_f1))
            nsml.report(summary=True, step=epoch, epoch_total=args.nb_epoch,
                        loss=train_loss.avg, f1=train_f1, val_loss=val_loss.avg, val_f1=val_f1)
            nsml.save(epoch)

            # Update learning rate
            lr_update(epoch, args, optimizer)