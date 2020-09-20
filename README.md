# KHD2020

## Getting Started
This repository provides everything necessary to train and evaluate a sinusitis classification model.\
This code is based on NSML environment.

Structure:
- ```datasets/```: data loader code
- ```network/```: network architecture definitions
- ```options/```: argument parser options
- ```utils/```: image processing code, miscellaneous helper functions, and training/evaluation core code
- ```train.py/```: code for model training and validation
- ```inference.py/```: code for model inference


#### Training and Testing
- To train a network, call:
```nsml run -d dataset_name -e train.py -a "--batch_size 32 --network resnet50"```

#### Performance
|   Network    |  Accuracy  |
| :----------: | :--------: |
|  ResNet-50   |            |