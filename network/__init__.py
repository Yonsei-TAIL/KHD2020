import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from efficientnet_pytorch import EfficientNet

def create_model(opt):
    # Load network
    if opt.network=='resnet18':
        net = resnet18(pretrained=True)
    elif opt.network=='resnet34':
        net = resnet34(pretrained=True)
    elif opt.network=='resnet50':
        net = resnet50(pretrained=True)
    elif opt.network=='efficientb0':
        net = EfficientNet.from_pretrained('efficientnet-b0')
    elif opt.network=='efficientb1':
        net = EfficientNet.from_pretrained('efficientnet-b1')
    else:
        raise ValueError("'resnet18', 'resnet34', 'resnet50', 'efficientb0', 'efficientb1' are supported now.")

    if opt.network.startswith("resnet"):
        net.fc = nn.Linear(net.fc.in_features, opt.n_classes)
    elif opt.network.startswith("efficient"):
        net._fc = nn.Linear(net._fc.in_features, opt.n_classes)

    # GPU settings
    if opt.use_gpu:
        net.cuda()
        if opt.ngpu > 1:
            net = torch.nn.DataParallel(net)
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            pretrained_dict = torch.load(opt.resume, map_location=torch.device('cpu'))
            model_dict = net.state_dict()

            match_cnt = 0
            mismatch_cnt = 0
            pretrained_dict_matched = dict()
            for k, v in pretrained_dict.items():
                if k in model_dict and v.size() == model_dict[k].size():
                    pretrained_dict_matched[k] = v
                    match_cnt += 1
                else:
                    mismatch_cnt += 1
                    
            model_dict.update(pretrained_dict_matched) 
            net.load_state_dict(model_dict)

            print("=> Successfully loaded weights from %s (%d matched / %d mismatched)" % (opt.resume, match_cnt, mismatch_cnt))

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    return net