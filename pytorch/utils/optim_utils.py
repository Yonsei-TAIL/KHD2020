import torch.nn as nn
import torch

def get_current_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def lr_update(epoch, args, optimizer):
    prev_lr = get_current_lr(optimizer)
    if (epoch + 1) in args.lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = (prev_lr * args.decay_ratio)
            print("LR Decay : %.7f to %.7f" % (prev_lr, prev_lr * 0.1))

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