# Korea Health Datathon 2020 Sinusitis Classification Solution

This repository is the 3rd place solution the sinusitis classifcation of [KHD2020](https://github.com/Korea-Health-Datathon/KHD2020) based on [sample code](https://github.com/KYBiMIL/KHD_2020).

**NOTE** : It is unable to run this code on your local machine because the challenge was based on [NSML](https://ai.nsml.navercorp.com/intro) infrastructure and the dataset was private. You have to customize the [data_loader.py](https://github.com/Yonsei-TAIL/KHD2020/blob/master/utils/data_loader.py) to run training code with your own dataset and specify the directory on [DATASET_PATH](https://github.com/Yonsei-TAIL/KHD2020/blob/master/utils/config.py#L8) argument.

## Getting Started
**Requirements** : run ```pip install -r requirements.txt```

#### [Pre-processing](https://github.com/Yonsei-TAIL/KHD2020/blob/master/utils/transform.py#L62-L84) : The specific process is descripted in the [notebook](https://github.com/Yonsei-TAIL/KHD2020/blob/master/preprocessing_example.ipynb).
1. Zero-padding to 300x600
2. Windowing
3. Background reduction
4. RoI crop to 224x224
5. Min-Max scaling

## Training Details :
We trained network using SGD optimizer with a momentum of 0.9 and decay of 0.3. However, we didn't apply weight decay on bias term. We used a decaying learning rate with a warm-up start with an initial learning rate 0.0005 and a minimum rate was 5e-6. We used a small batch size of 8 to increase training stability and trained the network for 60 epochs. The sinusitis dataset has a severe class imbalance, therefore, we used a class weights for loss function of 1:4:6:9. To avoid overfitting, we added dropout on the fully connected layer with 0.5 probability. We applied simple data augmentation techniques such as random rotation (-15~15 degress) and scaling (x0.85~1.15).  

#### Structure :
- ```model/```: network architecture definitions and training/validation core functions
- ```utils/```: image processing code, config argument parser, and data loading code
- ```main_pytorch.py```: code for model training and validation

#### NSML Environment
- Main command : ```nsml run -d 2020KHD_PNS -e main_pytorch.py -c 8 -a '--batch_size 64'```
- Submit command : ```nsml submit NSML_ID/2020KHD_PNS/SESSION_NUM CHECKPOINT_NUM```

#### Local Environment
- Main command : ```python main_pytorch.py --DATASET_PATH YOUR_DATASET_DIR --batch_size 64'```