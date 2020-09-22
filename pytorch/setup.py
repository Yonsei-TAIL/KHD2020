## INVALID NOW = nsml: nvcr.io/nvidia/pytorch:20.03-py3
from distutils.core import setup
import setuptools

print('setup_test.py is running...')

setup(name='PNS_SAMPLE',
      version='1.0',
      install_requires=['opencv-python',"imgaug","scikit-learn","efficientnet_pytorch","torch==1.1.0","torchvision==0.4.0"]
      ) ## install libraries, 'keras==xx.xx'  ## #nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2