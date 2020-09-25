import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from math import pi, cos, log, floor
from torch.autograd import Variable


def get_current_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def lr_update(epoch, args, optimizer):
    prev_lr = get_current_lr(optimizer)
    if (epoch + 1) in args.lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = (prev_lr * args.decay_ratio)
            print("LR Decay : %.7f to %.7f" % (prev_lr, prev_lr * 0.1))


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing_factor=0.1, class_weights=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        
        self.smoothing_factor = smoothing_factor
        self.class_weights = class_weights
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing_factor
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing_factor * smooth_loss
        
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduce=False)(inputs, targets)

        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

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