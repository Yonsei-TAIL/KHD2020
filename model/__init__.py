import os
import numpy as np

from utils.transform import ImagePreprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50

import nsml

def bind_model(model, args):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## test mode
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        X = ImagePreprocessing(data, args)
        X = np.array(X)
        X = np.expand_dims(X, axis=1)

        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(device)
            pred = model.forward(X)
            prob, pred_cls = torch.max(pred.softmax(dim=1), 1)
            pred_cls = pred_cls.tolist()

        print('Prediction done!\n Saving the result...')
        return pred_cls

    nsml.bind(save=save, load=load, infer=infer)

def load_resnet(resnet_type, pretrained=True):
    if resnet_type == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif resnet_type == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif resnet_type == 'resnet50':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError("resnet18, resnet34, and resnet50 are supported now.")

def load_model(args):
    #####   Model   #####
    if 'efficientnet' in args.network:
        model = EfficientNet.from_name(args.network)
        model._change_in_channels(1)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)

    elif 'resnet' in args.network:
        model = load_resnet(args.network, pretrained=True)
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3,
                            bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(model.fc.in_features, args.num_classes)
        )
        
    else:
        raise ValueError("resnet and efficientnet are only supported now.")
    
    return model