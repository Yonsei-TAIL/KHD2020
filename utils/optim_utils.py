from math import pi, cos

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

def get_current_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def load_optimizer(model, args):
    if not args.bias_decay:
        weight_params = []
        bias_params = []
        for n, p in model.named_parameters():
            if 'bias' in n:
                bias_params.append(p)
            else:
                weight_params.append(p)
        parameters = [{'params' : bias_params, 'weight_decay' : 0},
                      {'params' : weight_params}]
    else:
        parameters = model.parameters()

    if args.optim.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
  
    return optimizer

def load_loss_function(args):
    class_weights = torch.Tensor(args.class_weights[)
    criterion = nn.CrossEntropyLoss(class_weights)
    
    return criterion
        
class CosineWarmupLR(_LRScheduler):
    '''
    Cosine lr decay function with warmup.
    Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/optimizer/lr_scheduler.py
         https://github.com/Randl/MobileNetV3-pytorch/blob/master/cosine_with_warmup.py
    Lr warmup is proposed by
        `Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by
        `Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        iter_in_one_epoch (int): number of iterations in one epoch.
        epochs (int): number of epochs to train.
        lr_min (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        last_epoch (int): init iteration. In truth, this is last_iter
    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.
        cosine_iters (int): number of iterations of all cosine epochs.
    '''

    def __init__(self, optimizer, epochs, iter_in_one_epoch, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = lr_min
        self.niters = epochs * iter_in_one_epoch
        self.warmup_iters = iter_in_one_epoch * warmup_epochs
        self.cosine_iters = iter_in_one_epoch * (epochs - warmup_epochs)
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_iters) for base_lr in
                    self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * (
                    1 + cos(pi * (self.last_epoch - self.warmup_iters) / self.cosine_iters)) / 2) for base_lr in
                    self.base_lrs]