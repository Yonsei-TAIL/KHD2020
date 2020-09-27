# Korea Health Datathon 2020 Sinusitis Classification Solution

This repository is the 3rd place solution the sinusitis classifcation of [KHD2020](https://github.com/Korea-Health-Datathon/KHD2020) based on [sample code](https://github.com/KYBiMIL/KHD_2020).

**NOTE** : It is unable to run this code on your local machine because the challenge was based on [NSML](https://ai.nsml.navercorp.com/intro) infrastructure and the dataset was private. You have to customize the [data_loader.py](https://github.com/Yonsei-TAIL/KHD2020/blob/master/utils/data_loader.py) to run training code with your own dataset and specify the directory on DATASET_PATH argument.

Requirements: run ```pip install -r requirements.txt```

#### Steps
1. Pre-processing : 
2. 

Structure :
- ```model/```: network architecture definitions and training/validation core functions
- ```utils/```: image processing code, config argument parser, and data loading code
- ```main_pytorch.py/```: code for model training and validation

#### NSML Environment
- Main command : ```nsml run -d 2020KHD_PNS -e main_pytorch.py -c 8 -a '--batch_size 64'```
- Submit command : ```nsml submit NSML_ID/2020KHD_PNS/SESSION_NUM CHECKPOINT_NUM```

#### Local Environment
- Main command : ```python main_pytorch.py --DATASET_PATH YOUR_DATASET_DIR --batch_size 64'```