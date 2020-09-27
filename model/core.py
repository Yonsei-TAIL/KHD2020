from sklearn.metrics import f1_score

from utils import AverageMeter

import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(epoch, batch_train, device, optimizer, model, criterion, lr_fn, args):
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

        # cosing annealing
        lr_fn.step(epoch * len(batch_train) + i)

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
    
    return train_loss.avg, train_weighted_f1

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
    
    return val_loss.avg, val_weighted_f1
