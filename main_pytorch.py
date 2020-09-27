import cv2
import random
import os, sys
import numpy as np

from model import load_model, bind_model
from model.core import train_model, valid_model

from utils.data_loader import load_dataloader
from utils.config import ParserArguments
from utils.optim_utils import load_optimizer, load_loss_function, CosineWarmupLR

import torch
import torch.nn as nn

try:
    import nsml
    environ = 'nsml'
    print("NSML Environment.")
except:
    environ = 'local'
    print("Local Environment.")

# Seed
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == '__main__':
    # Argument
    args = ParserArguments()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model
    model = load_model(args)
    model.to(device)

    # Bind Model
    if environ=='nsml':
        bind_model(model, args)

    # Loss
    criterion = load_loss_function(args).to(device)

    # Optimizer
    optimizer = load_optimizer(model, args)

    ### Main ###
    if environ=='nsml' and args.pause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())

    elif args.mode == 'train':  ## for train mode
        print('Training start ...')
        batch_train, batch_val = load_dataloader(args)

        # Learning Rate Scheduler
        lr_fn = CosineWarmupLR(optimizer=optimizer, epochs=args.nb_epoch, iter_in_one_epoch=len(batch_train), lr_min=args.min_lr,
                               warmup_epochs=args.warmup_epoch)

        #####   Training and Validation loop   #####
        for epoch in range(args.nb_epoch):
            train_loss, train_f1 = train_model(epoch, batch_train, device, optimizer, model, criterion, lr_fn, args)
            val_loss, val_f1 = valid_model(epoch, batch_val, device, model, criterion, args)

            # Total summary
            print("  * Train loss = {:.4f} | Train F1 = {:.4f} | Val loss = {:.4f} | Val F1 = {:.4f}"\
                    .format(train_loss, train_f1, val_loss, val_f1))

            # Export result
            if environ == 'nsml':
                nsml.report(summary=True, step=epoch, epoch_total=args.nb_epoch,
                            loss=train_loss, f1=train_f1, val_loss=val_loss, val_f1=val_f1)
                nsml.save(epoch)
            else:
                torch.save(model.state_dict(),
                           os.path.join(args.exp, 'epoch_%03d_val_loss_%.4f_val_f1_%.4f.pth'%(epoch, val_loss, val_f1)))