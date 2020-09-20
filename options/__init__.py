import os
import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_option(print_option=True):    
    p = argparse.ArgumentParser(description='')

    # Data Directory
    p.add_argument('--data_root', default='data', type=str, help='root directory of dataset files.')
    
    # Data augmentation
    p.add_argument('--augmentation', default=True, type=str2bool, help='apply augmentation or not')
    p.add_argument('--rot_factor', default=30, type=float)
    p.add_argument('--scale_factor', default=0.15, type=float)
    p.add_argument('--flip', default='True', type=str2bool)
    p.add_argument('--trans_factor', default=0.1, type=float)

    # Input image
    p.add_argument('--in_res', default=256, type=int, help='input resolution using resize process')

    # Network
    p.add_argument('--network', default='resnet18', type=int, help='which network to use')
    p.add_argument('--n_classes', default=2, type=int, help='number of output channels (output mask types)')

    # Optimizer
    p.add_argument('--optim', default='Adam', type=str, help='RMSprop | SGD | Adam')
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--lr_decay_epoch', default='10,15', type=str, help="decay epochs with comma (ex - '20,40,60')")
    p.add_argument('--lr_warmup_epoch', default=3, type=int)
    p.add_argument('--momentum', default=0, type=float, help='momentum')
    p.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    p.add_argument('--no_bias_decay', default='True', type=str2bool, help='weight decay for bias')

    # Hyper-parameter
    p.add_argument('--batch_size', default=16, type=int, help='use 1 batch size in 3D training.')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--max_epoch', default=20, type=int)

    # Loss function
    p.add_argument('--loss', default='ce', type=str)
    p.add_argument('--weights', default=0.5, type=float, help='class weights for positive label')

    # Resume trained network
    p.add_argument('--resume', default='', type=str, help="pth file path to resume")

    # Resource option
    p.add_argument('--workers', default=10, type=int, help='#data-loading worker-processes')
    p.add_argument('--use_gpu', default="True", type=str2bool, help='use gpu or not (cpu only)')
    p.add_argument('--gpu_id', default="0", type=str)

    # Output directory
    p.add_argument('--exp', default='exp', type=str, help='checkpoint dir.')


    opt = p.parse_args()
    
    # Make output directory
    if not os.path.exists(opt.exp):
        os.makedirs(opt.exp)

    # GPU Setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id

    opt.ngpu = len(opt.gpu_id.split(","))

    # lr decay setting
    if ',' in opt.lr_decay_epoch:
        opt.lr_decay_epoch = opt.lr_decay_epoch.split(',')
        opt.lr_decay_epoch = [int(epoch) for epoch in opt.lr_decay_epoch]

    if print_option:
        print("\n==================================== Options ====================================\n")
    
        print('   Data root : %s' % (opt.data_root))
        print()
        print('   Data augmentation : %s' % (opt.augmentation))
        print('   Data input size : Resized to (%d,%d)' % (opt.in_res,opt.in_res))
        print()
        print('   #Classes : %d' % (opt.n_classes))
        print()
        print('   Optimizer : %s' % (opt.optim))
        print('   Loss function : %s' % opt.loss)
        print('   Batch size : %d' % opt.batch_size)
        print('   Max epoch : %d' % opt.max_epoch)
        print('   Learning rate : %s (linear warm-up until %s / decay at %s)' % (opt.lr, opt.lr_warmup_epoch, opt.lr_decay_epoch))
        print()
        print('   Resume pre-trained weights path : %s' % opt.resume)
        print('   Output dir : %s' % opt.exp)
        print()
        print('   GPU ID : %s' % opt.gpu_id)
        print('   #Workers : %s' % opt.workers)
        print('   pytorch version: %s (CUDA : %s)' % (torch.__version__, torch.cuda.is_available()))
        print("\n=================================================================================\n")

    return opt