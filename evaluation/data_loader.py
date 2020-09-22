import os, sys
import time
import numpy as np
import cv2

import nsml
from nsml.constants import DATASET_PATH


def image_padding(img_whole):
    img = np.zeros((600,600))
    h, w = img_whole.shape

    if (600 - h) != 0:
        gap = int((600 - h)/2)
        img[gap:gap+h,:] = img_whole
    elif (600 - w) != 0:
        gap = int((600 - w)/2)
        img[:,gap:gap+w] = img_whole
    else:
        img = img_whole

    return img

def LoadTestData(imdir):
    print('Loading test data...')
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    fn = []
    dir_lr = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        # zero padding
        img_whole = image_padding(img_whole)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, :w_]
        r_img = img_whole[:, w_:2*w_]
        r_img = cv2.flip(r_img, 0) # Flip Right Image to Left

        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls == '0' or l_cls == '1' or l_cls == '2' or l_cls == '3':
            fn.append(p);   dir_lr.append('l'); img.append(l_img);      lb.append(l_cls)
        if r_cls == '0' or r_cls == '1' or r_cls == '2' or r_cls == '3':
            fn.append(p);   dir_lr.append('r'); img.append(r_img);      lb.append(r_cls)
    print(len(img), 'test data with label 0-3 loaded!')
    return fn, dir_lr, img, lb


def MakePredFile(fn, dir_lr, labels, res, output_file):
    lines = []
    for i, f in enumerate(fn):
        line = ','.join([fn[i], dir_lr[i], labels[i], res[i]])
        #line = ','.join([fn[i], dir_lr[i], res[i]])
        lines.append(line)

    print('Writing output file ...')
    with open(output_file, 'wt') as file_writer:
        file_writer.write('\n'.join(lines))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

    return True


def feed_infer(output_file, infer_func):
    fn, dir_lr, images, labels = LoadTestData(os.path.join(DATASET_PATH, 'test'))
    res = infer_func(images)
    res = [str(r) for r in res]
    MakePredFile(fn, dir_lr, labels, res, output_file)