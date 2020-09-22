#nsml: pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
from distutils.core import setup
import setuptools

print('setup_test.py is running...')

setup(name='PNS_SAMPLE',
      version='1.0',
      install_requires=['opencv-python',"imgaug","scikit-learn","efficientnet_pytorch","libgl1-mesa-glx"]
      ) ## install libraries, 'keras==xx.xx'  ## #nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2