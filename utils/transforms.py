import cv2
import torch
import numpy as np
import SimpleITK as sitk
from scipy.misc import imresize
from random import uniform
from imgaug import augmenters as iaa

def ResizeImage(img, new_shape):
    return imresize(img[0], new_shape, mode='L')[None, ...]

def center_crop(img_array, x_size, y_size):
    y, x = img_array.shape[-2:]

    if (y < y_size) or (x < x_size):
        return img_array
        
    x_start = (x//2) - (x_size//2)
    y_start = (y//2) - (y_size//2)
    
    img_crop = img_array[...,
                    y_start : y_start + y_size,
                    x_start : x_start + x_size]

    return img_crop

def augment_imgs(imgs, rot_factor, scale_factor, trans_factor, flip):
    rot_factor = uniform(-rot_factor, rot_factor)
    scale_factor = uniform(1-scale_factor, 1+scale_factor)
    trans_factor = [int(imgs.shape[1]*uniform(-trans_factor, trans_factor)),
                    int(imgs.shape[2]*uniform(-trans_factor, trans_factor))]

    # 2D input
    if np.ndim(imgs) == 3:
        seq = iaa.Sequential([
                iaa.Affine(
                    translate_px={"x": trans_factor[0], "y": trans_factor[1]},
                    scale=(scale_factor, scale_factor),
                    rotate=rot_factor
                )
            ])

        seq_det = seq.to_deterministic()

        imgs = seq_det.augment_images(imgs)

        if flip and uniform(0, 1) > 0.5:
            imgs = np.flip(imgs, 2).copy()

    # 3D input
    elif np.ndim(imgs) == 4:
        pass

    return imgs