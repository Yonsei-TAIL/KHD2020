import torch
from torch.utils.data import Dataset

import os
import numpy as np
from glob import glob
import SimpleITK as sitk

from utils.transforms import ResizeImage, center_crop, mask_binarization, augment_imgs_and_masks

class SinusitisDataset(Dataset):
    def __init__(self, data_root, opt, is_Train=True, augmentation=False):
        super(SinusitisDataset, self).__init__()
        '''
        1. Change image loading code according to image data extension.
        2. Calculate MEAN and STD for z-score normalization 
        '''
        self.data_list = glob(os.path.join(data_root, 'train' if is_Train else 'valid', '*.png'))
        self.len = len(self.data_list)

        self.augmentation = augmentation
        self.rot_factor = opt.rot_factor
        self.scale_factor = opt.scale_factor
        self.flip = opt.flip
        self.trans_factor = opt.trans_factor

        self.in_res = opt.in_res

        self.is_Train = is_Train

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label = '''specify'''

        imgs = [ResizeImage(img, (self.in_res, self.in_res)) for img in imgs]
        imgs = [img[None, ...] for img in imgs]

        return imgs, label
        
    def _load_img(self, img_path):
        '''
        specify img loading code
        '''
        return img

    def __len__(self):
        return self.len