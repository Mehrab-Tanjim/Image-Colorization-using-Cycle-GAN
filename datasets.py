import glob
import random
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input
import numpy as np # linear algebra
from matplotlib import cm
import time


import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import scipy.io as io
import scipy.misc as misc
import glob
import csv
from skimage import color
#from transform import ReLabel, ToLabel, ToSP, Scale


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ImageDataset(Dataset):
     
    def get_rbg_from_lab(self, gray_imgs, ab_imgs, n = 10):
        imgs = np.zeros((n, 224, 224, 3))
        imgs[:, :, :, 0] = gray_imgs[0:n:]
        imgs[:, :, :, 1:] = ab_imgs[0:n:]

        imgs = imgs.astype("uint8")

        imgs_ = []
        for i in range(0, n):
            imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

        imgs_ = np.array(imgs_)

#         print(imgs_.shape)

        return imgs_
    
    def pipe_line_img(self, gray_scale_imgs, batch_size = 100, preprocess_f = preprocess_input):
        imgs = np.zeros((batch_size, 224, 224, 3))
        for i in range(0, 3):
            imgs[:batch_size, :, :,i] = gray_scale_imgs[:batch_size]
        return preprocess_f(imgs)


    
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        print("in here")
        
        images_gray = np.load(root+'A/gray_scale.npy')
        images_lab = np.load(root+'B/ab1.npy')
        
        self.files_A = self.pipe_line_img(images_gray, batch_size = images_gray.shape[0]).transpose(0, 3, 1, 2)
        self.files_B = preprocess_input(self.get_rbg_from_lab(gray_imgs = images_gray, ab_imgs = images_lab, n = images_gray.shape[0])).transpose(0, 3, 1, 2)
        print("rearranged",self.files_A.shape)
   
    def __getitem__(self, index):
#         print(self.files_A.shape)
#         item_A = self.transform(Image.fromarray(np.uint8(cm.gist_earth(self.files_A[index])*255))) #TODO
#         item_B = self.transform(Image.fromarray(np.uint8(cm.gist_earth(self.files_B[index])*255))) #TODO

        item_A = self.files_A[index]
        item_B = self.files_B[index]

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    