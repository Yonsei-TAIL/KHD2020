import os
import argparse

def ParserArguments():
    args = argparse.ArgumentParser()

    # Directory Setting 
    args.add_argument('--DATASET_PATH', type=str, help='specify your own dataset')
    args.add_argument('--exp', type=str, default='./exp', help='output directory')

    # Hyperparameters Setting 
    args.add_argument('--nb_epoch', type=int, default=60, help='number of epochs (default=60)')
    args.add_argument('--batch_size', type=int, default=8, help='batch size (default=8)')
    args.add_argument('--num_classes', type=int, default=4, help='number of classes (default=4)')
    
    # Pre-processing
    args.add_argument('--img_size', type=int, default=224, help='input size (default=224)')
    args.add_argument('--w_min', type=int, default=50, help='window min value (default=50)')
    args.add_argument('--w_max', type=int, default=180, help='window max value (default=180)')

    # Optimization Settings
    args.add_argument('--learning_rate', type=float, default=5e-4, help='initial learning rate (default=5e-4)')
    args.add_argument('--optim', type=str, default='SGD', help='optimizer (default=SGD)')
    args.add_argument('--momentum', type=float, default=0.9, help='momentum (default=0.9)')
    args.add_argument('--wd', type=float, default=3e-2, help='weight decay of optimizer (default=0.03)')
    args.add_argument('--bias_decay', action='store_true', help='apply weight decay on bias (default=False)')
    args.add_argument('--warmup_epoch', type=int, default=5, help='learning rate warm-up epoch (default=5)')
    args.add_argument('--min_lr', type=float, default=5e-6, help='minimum learning rate setting of cosine annealing (default=5e-6)')
    args.add_argument('--class_weights', type=str, default='1,4,6,9', help='class weights for loss function (default=1,4,6,9)')

    # Network
    args.add_argument('--network', type=str, default='resnet34', help='classifier network (default=resnet34)')
    args.add_argument('--resume', type=str, default='', help='resume pre-trained weights')
    args.add_argument('--dropout', type=float, default=0.5, help='dropout rate of FC layer (default=0.5)')

    # Augmentation
    args.add_argument('--augmentation', type=str, default='light', help="apply light or heavy augmentation (default=light)")          
    args.add_argument('--rot_factor', type=float, default=15, help='max rotation degree of augmentation (default=15)')
    args.add_argument('--scale_factor', type=float, default=0.15, help='max scaling factor of augmentation (default=0.15)')

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = args.parse_args()
    
    # Change string type to list type
    args.lr_decay_epoch = list(map(int, args.lr_decay_epoch.split(',')))
    args.class_weights = list(map(float, args.class_weights.split(',')))

    # Make Output Directory
    os.makedirs(args.exp, exist_ok=True)

    return args