import cv2
import random
import os, sys
import argparse
import numpy as np

from utils.data_loader import load_dataloader
from utils.model import load_model, bind_model, train_model, valid_model, Ensemble
from utils.optim_utils import lr_update, load_optimizer, CosineWarmupLR, LabelSmoothingCrossEntropy

import torch
import torch.nn as nn

import nsml

import copy


def ParserArguments():
    args = argparse.ArgumentParser()

    # Setting Hyperparameters
    args.add_argument('--nb_epoch', type=int, default=60)       # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)      # 분류될 클래스 수는 4개
    args.add_argument('--stack_channels', action='store_true')   # 2가지 window로 만들어낸 이미지를 input channel로 쌓아 3-channel로
    
    # Pre-processing
    args.add_argument('--img_size', type=int, default=224) # input crop image size
    args.add_argument('--w_min', type=int, default=50) # Window mind
    args.add_argument('--w_max', type=int, default=180) # Window max
    args.add_argument('--zscore', action='store_true', help="apply ImageNet-based Z-score normalization")
    args.add_argument('--balancing_method', type=str, default='weights', help="'aug' : augmentation / 'weights' : class_weights")

    # Optimization Settings
    args.add_argument('--learning_rate', type=float, default=5e-4)  # learning rate 설정
    args.add_argument('--lr_decay_epoch', type=str, default='30,45')  # learning rate decay epoch
    args.add_argument('--decay_ratio', type=float, default=0.1)  # learning rate decay epoch
    args.add_argument('--optim', type=str, default='sgd')  # Optimizer
    args.add_argument('--momentum', type=float, default=0.9)  # Momentum
    args.add_argument('--wd', type=float, default=3e-2)  # Weight decay
    args.add_argument('--bias_decay', action='store_true')  # 선언 시 bias에도 weight decay 적용
    args.add_argument('--cosine_annealing', action='store_true')  # 선언 시 bias에도 weight decay 적용
    args.add_argument('--warmup_epoch', type=int, default=5)  # 선언 시 bias에도 weight decay 적용
    args.add_argument('--min_lr', type=float, default=0.000005)  # Weight decay
    args.add_argument('--smoothing_factor', type=float, default=0.1)  # Weight decay

    # Ensemble
    args.add_argument('--ensemble', action='store_true')  # True, if Ensemble
    args.add_argument('--num_models', type=int, default=1)  # Ensemble model numbers

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
    if args.ensemble:
        model = Ensemble(args)
    else:
        model = load_model(args)
    model.to(device)

    # Bind Model
    bind_model(model, args)

    # Loss
    if args.balancing_method == 'weights':
        class_weights = torch.Tensor([1,4,6,9])

        if args.smoothing_factor > 0:
            criterion = LabelSmoothingCrossEntropy().to(device)        
        else:
            criterion = nn.CrossEntropyLoss(args.smoothing_factor, class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = load_optimizer(model, args)


    if args.pause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())


    if args.mode == 'train' and args.ensemble:  # for Ensemble train mode
        print('Training for Ensemble start ...')
        models_weight_list = []
        epoch_patient = 15

        for n_model, RANDOM_SEED in zip(range(args.num_models), random.sample(range(0,100), args.num_models)):
            print("Start %d_model"%n_model)
            # Seed
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)

            model_part = load_model(args).to(device)
            # Optimizer
            optimizer = load_optimizer(model_part, args)

            batch_train, batch_val = load_dataloader(args)
            second_best_model_weight, best_model_weight = None, None
            best_loss = 10000
            second_best_loss = 10000
            
            #####   Training loop   #####
            epoch_cnt, n_decay = 0, 0
            for epoch in range(args.nb_epoch):
                train_loss, train_f1 = train_model(epoch, batch_train, device, optimizer, model_part, criterion, args)
                val_loss, val_f1 = valid_model(epoch, batch_val, device, model_part, criterion, args)

                if val_loss.avg < second_best_loss:
                    second_best_loss = val_loss.avg
                    second_best_model_weight = copy.deepcopy(model_part.state_dict())

                if val_loss.avg < best_loss:
                    epoch_cnt = 0
                    second_best_loss = best_loss
                    best_loss = val_loss.avg
                    best_model_weight = copy.deepcopy(model_part.state_dict())
                else:
                    epoch_cnt += 1

                # total summary
                print("  * Train loss = {:.4f} | Train F1 = {:.4f} | Val loss = {:.4f} | Val F1 = {:.4f}" \
                      .format(train_loss.avg, train_f1, val_loss.avg, val_f1))

                nsml.report(summary=True, step=epoch, epoch_total=args.nb_epoch,
                            loss=train_loss.avg, f1=train_f1, val_loss=val_loss.avg, val_f1=val_f1)

                # Update learning rate
                # if (epoch_cnt > epoch_patient) and (epoch > 30):
                #     n_decay += 1
                #     if n_decay == 4:
                #         print("Early Stopping...\n")
                #         break

                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] *= 0.2
                #         print("LR Decay from %.7f to %.7f" % (param_group['lr']*5, param_group['lr']))
                #     epoch_cnt = 0
                lr_update(epoch, args, optimizer)

            models_weight_list.append([best_model_weight, second_best_model_weight])

        cnt = 0
        for i in models_weight_list[0]:
            for j in models_weight_list[1]:
                for k in models_weight_list[2]:
                    model._load_trained_networks([i,j,k])
                    nsml.save(cnt)
                    cnt += 1


    elif args.mode == 'train':  ## for train mode
        # Seed
        RANDOM_SEED = 1234
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        print('Training start ...')
        batch_train, batch_val = load_dataloader(args)

        # Learning Rate Scheduler
        lr_fn = CosineWarmupLR(optimizer=optimizer, epochs=60, iter_in_one_epoch=len(batch_train), lr_min=args.min_lr,
                               warmup_epochs=args.warmup_epoch)


        #####   Training loop   #####
        for epoch in range(args.nb_epoch):
            train_loss, train_f1 = train_model(epoch, batch_train, device, optimizer, model, criterion, lr_fn, args)
            val_loss, val_f1 = valid_model(epoch, batch_val, device, model, criterion, args)

            # total summary
            print("  * Train loss = {:.4f} | Train F1 = {:.4f} | Val loss = {:.4f} | Val F1 = {:.4f}"\
                    .format(train_loss.avg, train_f1, val_loss.avg, val_f1))
            nsml.report(summary=True, step=epoch, epoch_total=args.nb_epoch,
                        loss=train_loss.avg, f1=train_f1, val_loss=val_loss.avg, val_f1=val_f1)
            nsml.save(epoch)

            # Update learning rate
            lr_update(epoch, args, optimizer)
