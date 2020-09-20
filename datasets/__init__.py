import torch
from torch.utils.data import DataLoader
from datasets.Sinusitis import SinusitisDataset

def get_dataloader(opt):
    trn_dataset = SinusitisDataset(opt.data_root, opt, is_Train=True, augmentation=opt.augmentation)
    val_dataset = SinusitisDataset(opt.data_root, opt, is_Train=False, augmentation=False)

    train_dataloader = DataLoader(trn_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)

    valid_dataloader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                num_workers=opt.workers)
    
    return train_dataloader, valid_dataloader