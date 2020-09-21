
import os, sys
import argparse
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM



IMSIZE = 120, 60
VAL_RATIO = 0.2

# Seed
RANDOM_SEED = 44
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## test mode
        X = ImagePreprocessing(data)
        X = np.array(X)
        X = np.expand_dims(X, axis=1)
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(device)
            pred = model.forward(X)
            prob, pred_cls = torch.max(pred, 1)
            pred_cls = pred_cls.tolist()
            #pred_cls = pred_cls.data.cpu().numpy()
        print('Prediction done!\n Saving the result...')
        return pred_cls

    nsml.bind(save=save, load=load, infer=infer)


def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, :w_]
        r_img = img_whole[:, w_:2*w_]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img);      lb.append(int(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img);      lb.append(int(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb


def ImagePreprocessing(img):
    # 자유롭게 작성
    h, w = IMSIZE
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        img[i] = tmp
    print(len(img), 'images processed!')
    return img


def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=10)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-5)  # learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.pause, config.mode


class SampleModelTorch(nn.Module):
    def __init__(self, num_classes=4):
        super(SampleModelTorch, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256 * 30 * 15, 2048),
                                nn.Linear(2048, 128),
                                nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

'''
class PNSDataset(Dataset):
    def __init(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
'''

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    print(GPU_NUM)
    nb_epoch, batch_size, num_classes, learning_rate, ifpause, ifmode = ParserArguments(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #####   Model   #####
    model = SampleModelTorch(num_classes)
    #model.double()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bind_model(model)

    if ifpause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ## for train mode
        print('Training start ...')
        # 자유롭게 작성
        images, labels = DataLoad(imdir=os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images)
        images = np.array(images)
        images = np.expand_dims(images, axis=1)
        labels = np.array(labels)

        dataset = TensorDataset(torch.from_numpy(images).float(), torch.from_numpy(labels).long())
        subset_size = [len(images) - int(len(images) * VAL_RATIO),int(len(images) * VAL_RATIO)]
        tr_set, val_set = random_split(dataset, subset_size)
        batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        batch_val = DataLoader(val_set, batch_size=1, shuffle=False)

        #####   Training loop   #####
        STEP_SIZE_TRAIN = len(images) // batch_size
        print('\n\n STEP_SIZE_TRAIN= {}\n\n'.format(STEP_SIZE_TRAIN))
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print('Model fitting ...')
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))
            a, a_val, tp, tp_val = 0, 0, 0, 0
            for i, (x_tr, y_tr) in enumerate(batch_train):
                x_tr, y_tr = x_tr.to(device), y_tr.to(device)
                optimizer.zero_grad()
                pred = model(x_tr)
                loss = criterion(pred, y_tr)
                loss.backward()
                optimizer.step()
                prob, pred_cls = torch.max(pred, 1)
                a += y_tr.size(0)
                tp += (pred_cls == y_tr).sum().item()

            with torch.no_grad():
                for j, (x_val, y_val) in enumerate(batch_val):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, y_val)
                    prob_val, pred_cls_val = torch.max(pred_val, 1)
                    a_val += y_val.size(0)
                    tp_val += (pred_cls_val == y_val).sum().item()

            acc = tp / a
            acc_val = tp_val / a_val
            print("  * loss = {}\n  * acc = {}\n  * loss_val = {}\n  * acc_val = {}".format(loss.item(), acc, loss_val.item(), acc_val))
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=loss.item(), acc=acc, val_loss=loss_val.item(), val_acc=acc_val)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f\n' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))