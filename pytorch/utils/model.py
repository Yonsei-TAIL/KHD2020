import os
import numpy as np
from sklearn.metrics import f1_score

from utils import AverageMeter
from utils.transform import ImagePreprocessing

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34

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
        if not args.stack_channels:
            X = np.expand_dims(X, axis=1)
        ##### DO NOT CHANGE ORDER OF TEST DATA #####

        model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(device)
            pred = model.forward(X)
            prob, pred_cls = torch.max(pred, 1)
            pred_cls = pred_cls.tolist()
            #pred_cls = pred_cls.data.cpu().numpy()
        print('Prediction done!\n Saving the result...')
        return pred_cls

    nsml.bind(save=save, load=load, infer=infer)

def load_model(args):
    #####   Model   #####
    if 'efficientnet' in args.network:
        model = EfficientNet.from_name(args.network)
        if not args.stack_channels:
            model._change_in_channels(1)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)

    elif args.network == 'resnet18':
        model = resnet18(pretrained=False)
        if not args.stack_channels:
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    elif args.network == 'resnet34':
        model = resnet34(pretrained=False)
        if not args.stack_channels:
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(model.fc.in_features, args.num_classes)
        )
    
    return model

def train_model(epoch, batch_train, device, optimizer, model, criterion, args):
    model.train()
    print('Model fitting ...')
    print('epoch = {} / {}'.format(epoch + 1, args.nb_epoch))
    print('check point = {}'.format(epoch))

    ## Training
    true_labels = []
    pred_labels = []
    train_loss = AverageMeter()
    for i, (x_tr, y_tr) in enumerate(batch_train):
        optimizer.zero_grad()
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        
        pred = model(x_tr)
        loss = criterion(pred, y_tr)

        loss.backward()
        optimizer.step()

        _, pred_cls = torch.max(pred, 1)

        train_loss.update(loss.item(), len(x_tr))
        true_labels.extend(list(y_tr.cpu().numpy().astype(int)))
        pred_labels.extend(list(pred_cls.cpu().numpy().astype(int)))

        if i>0 and i%10 == 0:
            print("  * Iter Loss [{:d}/{:d}] loss = {}".format(i+1, len(batch_train), train_loss.avg))

    # train performance
    class0_f1, class1_f1, class2_f1, class3_f1 = f1_score(true_labels, pred_labels, average=None)
    train_weighted_f1 = (class0_f1 + class1_f1*2 + class2_f1*3 + class3_f1*4) / 10.
    print("  * Train Class1 F1= {:.2f} | Class2 F1 = {:.2f} | Class3 F1 = {:.2f} | Class4 F1 = {:.2f} | Weighted F1 = {:.2f}"\
        .format(class0_f1, class1_f1, class2_f1, class3_f1, train_weighted_f1))
    
    return train_loss, train_weighted_f1

def valid_model(epoch, batch_val, device, model, criterion, args):
    model.eval()

    val_loss = AverageMeter()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for j, (x_val, y_val) in enumerate(batch_val):
            x_val, y_val = x_val.to(device), y_val.to(device)

            pred_val = model(x_val)
            loss_val = criterion(pred_val, y_val)

            _, pred_cls_val = torch.max(pred_val, 1)

            val_loss.update(loss_val.item(), len(x_val))
            true_labels.extend(list(y_val.cpu().numpy().astype(int)))
            pred_labels.extend(list(pred_cls_val.cpu().numpy().astype(int)))

    # validation performance
    class0_f1, class1_f1, class2_f1, class3_f1 = f1_score(true_labels, pred_labels, average=None)
    val_weighted_f1 = (class0_f1 + class1_f1*2 + class2_f1*3 + class3_f1*4) / 10.
    print("  * Valid Class1 F1= {:.2f} | Class2 F1 = {:.2f} | Class3 F1 = {:.2f} | Class4 F1 = {:.2f} | Weighted F1 = {:.2f}"\
        .format(class0_f1, class1_f1, class2_f1, class3_f1, val_weighted_f1))
    
    return val_loss, val_weighted_f1