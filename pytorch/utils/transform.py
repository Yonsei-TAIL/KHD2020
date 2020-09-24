import numpy as np

def image_padding(img_whole):
    img = np.zeros((600,300))
    h, w = img_whole.shape

    if (600 - h) != 0:
        gap = int((600 - h)/2)
        img[gap:gap+h,:] = img_whole
    elif (300 - w) != 0:
        gap = int((300 - w)/2)
        img[:,gap:gap+w] = img_whole
    else:
        img = img_whole

    return img

def image_windowing(img, w_min=50, w_max=180):
    img_w = img.copy()

    img_w[img_w < w_min] = w_min
    img_w[img_w > w_max] = w_max

    return img_w
    
def image_bg_reduction(img):
    img_wo_bg = img.copy()
    
    if np.ndim(img) == 3:
        for d in range(3):
            img_wo_bg[d] = image_bg_reduction(img[d])
    else:
        img_wo_bg[:250] = np.min(img)
        img_wo_bg[500:] = np.min(img)
        img_wo_bg[:, :60] = np.min(img)
        img_wo_bg[:, -60:] = np.min(img)

    return img_wo_bg

def image_roi_crop(img, img_size):
    half_size = img_size // 2
    return img[..., 350-half_size:350+half_size,
               150-half_size:150+half_size].copy()

def image_minmax(img):
    img_minmax = img.copy()
    
    if np.ndim(img) == 3:
        for d in range(3):
            img_minmax[d] = image_minmax(img[d])
    else:
        img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
        
    return img_minmax

def ImagePreprocessing(img, args):
    # 자유롭게 작성
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        # Zero-padding
        im = image_padding(im)
        
        # Windowing
        if args.stack_channels:
            im_w1 = image_windowing(im, args.w_min, args.w_max)
            im_w2 = image_windowing(im, 100, 200)
            im = np.stack([im, im_w1, im_w2], 0)
        else:
            im = image_windowing(im, args.w_min, args.w_max)

        # Background reduction
        im = image_bg_reduction(im)
        
        # RoI crop
        im = image_roi_crop(im, args.img_size)
        
        # Min-Max scaling
        im = image_minmax(im)

        img[i] = im
        
    print(len(img), 'images processed!')
    return img