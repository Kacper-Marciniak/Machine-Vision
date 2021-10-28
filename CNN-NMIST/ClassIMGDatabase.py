import torch
 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
 
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth = 120)

from torch.utils.data import Dataset, DataLoader
 
import numpy as np
import os
import cv2

import PIL

img_size = (28,28)

TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
    ])

#PATH_TRAIN = r'D:/DATABASE/TRAIN/'
#PATH_TEST = r'D:/DATABASE/TEST/'

class ImageDatasetTrain(Dataset):
    def __init__(self, PATH):

        self.imgs_path = PATH
        self.img_size = img_size
        self.data = []

        #Load data to self.data
        for filename in os.listdir(self.imgs_path):
            if os.path.isdir(os.path.join(self.imgs_path, filename)):
                # skip directories
                continue
            with open(os.path.join(self.imgs_path, filename), 'r') as f:
                img_path = f.readline().rstrip() 
                label = f.readline().rstrip() 
                self.data.append([img_path, int(label)])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path, label = self.data[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Change to greyscale
        
        img_np = np.array(img)
        img_np = np.uint8(img_np) #Change to 8 bit
        
        pilInput = PIL.Image.fromarray(img_np)
        
        img_tensor = TRANSFORM(pilInput)     

        class_id = torch.tensor(label)
        return img_tensor, class_id

class ImageDatasetTest(Dataset):
    def __init__(self, PATH):
        
        self.imgs_path = PATH
        self.img_size = img_size
        self.data = []
        
        #Load data to self.data
        for filename in os.listdir(self.imgs_path):
            if os.path.isdir(os.path.join(self.imgs_path, filename)):
                # skip directories
                continue
            with open(os.path.join(self.imgs_path, filename), 'r') as f:
                img_path = f.readline().rstrip() 
                label = f.readline().rstrip() 
                self.data.append([img_path, int(label)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path, label = self.data[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Change to greyscale
        
        img_np = np.array(img)
        img_np = np.uint8(img_np) #Change to 8 bit
        
        pilInput = PIL.Image.fromarray(img_np)
        img_tensor = TRANSFORM(pilInput)     

        class_id = torch.tensor(label)
        return img_tensor, class_id