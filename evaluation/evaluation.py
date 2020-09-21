import os
import argparse
import numpy as np



def GetF1score(y, y_pred, target):
    tp = 0
    fp = 0
    fn = 0
    for i, y_hat in enumerate(y_pred):
        if (y[i] == target) and (y_hat == target):
            tp += 1
        if (y[i] == target) and (y_hat != target):
            fn += 1
        if (y[i] != target) and (y_hat == target):
            fp += 1
    f1s = tp / ( tp + (fp + fn)/2 )
    return f1s

def CategoricalF1Score(outfilepath, num_classes):
    res_arr = np.loadtxt(outfilepath, dtype='str')
    y, y_pred = [], []
    for i, res in enumerate(res_arr):
        r1, r2, r3, r4 = res.split(',')
        y.append(r3)
        y_pred.append(r4)
    F1scores = []
    for t in range(num_classes):
        F1scores.append(GetF1score(y, y_pred, str(t)))
    return F1scores


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str, default='pred.txt')
    config = args.parse_args()
    outfilepath = config.prediction
    num_classes = 4

    F1scores = CategoricalF1Score(outfilepath, num_classes)
    SCORE = 0
    for i, s in enumerate(F1scores):
        SCORE += (i + 1) * s / 10
    SCORE = round(SCORE, 10)
    print(SCORE)