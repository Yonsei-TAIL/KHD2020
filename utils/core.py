import os
import torch
import numpy as np
from glob import glob

import torch
from torch.autograd import Variable

from utils import AverageMeter

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")
    net.train()

    losses, acc_results = AverageMeter(), AverageMeter()
    for it, (img, label) in enumerate(dataset_trn):
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img, label = torch.Tensor(img).float(), torch.Tensor(label).float()
        if opt.use_gpu:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, label)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Calculate Accuracy
        n_imgs = img.size(0)
        _, pred_label = torch.max(pred.data, 1)
        acc = (pred == pred_label).sum().cpu().item() / n_imgs
        acc_results.update(acc, n_imgs)

        # Stack Results
        losses.update(loss.item(), n_imgs)

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Acc %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, acc_results.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Acc %.4f\n"
        % (epoch+1, opt.max_epoch, losses.avg, acc_results.avg))


def validate(dataset_val, net, criterion, optimizer, epoch, opt, best_acc, best_epoch):
    print("Start Evaluation...")
    net.eval()

    # Result containers
    losses, acc_results = AverageMeter(), AverageMeter()
    for it, (img, label) in enumerate(dataset_val):
        # Load Data
        img, label = torch.Tensor(img).float(), torch.Tensor(label).float()
        if opt.use_gpu:
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, label)

        # Calculate Accuracy
        n_imgs = img.size(0)
        _, pred_label = torch.max(pred.data, 1)
        acc = (pred == pred_label).sum().cpu().item() / n_imgs
        acc_results.update(acc, n_imgs)

        # Stack Results
        losses.update(loss.item(), n_imgs)

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Acc %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_val), losses.avg, acc_results.avg))

    print(">>> Epoch[%3d/%3d] | Validation Loss : %.4f | Acc %.4f\n"
        % (epoch+1, opt.max_epoch, losses.avg, acc_results.avg))

    # Update Result
    if acc_results.avg > best_acc:
        print('Best Score Updated...')
        best_acc = acc_results.avg
        best_epoch = epoch

        # Remove previous weights pth files
        for path in glob('%s/*.pth' % opt.exp):
            os.remove(path)

        model_filename = '%s/epoch_%04d_acc%.4f_loss%.8f.pth' % (opt.exp, epoch+1, best_acc, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: Accuracy %.8f in %3d epoch\n' % (best_acc, best_epoch+1))
    return best_acc, best_epoch