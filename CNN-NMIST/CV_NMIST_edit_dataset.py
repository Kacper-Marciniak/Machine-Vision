from numpy.lib import type_check
import torch
import torchvision
import torchvision.transforms as transforms
 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
 
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth =120)
 
import numpy as np
 
import os
import sys

import cv2

import PIL

def processDataSet(set):
    i = 0
    new_set = list()

    for img, label in set:
        
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = cv2.resize(img_np, (100,100), interpolation = cv2.INTER_AREA)
        img_np *= 255
        img_np = np.uint8(img_np)
        
        _, img_tresh = cv2.threshold(img_np, 5, 255, cv2.THRESH_OTSU) # Apply treshholding

        contours, hierarchy = cv2.findContours(img_tresh , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) # Find contours
        c = max(contours, key = cv2.contourArea)
        rect = cv2.boundingRect(c)

        r_size = int(max([int(rect[2]),int(rect[3])]))
        x = int(rect[0] - (r_size-rect[2])/2) # Calculate new offset values (x,y)
        if x < 0: x = 0
        y = int(rect[1] - (r_size-rect[3])/2)
        if y < 0: y = 0

        img_processed = img_np[ y : y+r_size, x : x+r_size]
        img_processed = cv2.resize(img_processed, (28,28), interpolation = cv2.INTER_AREA)\

        new_set.append((img_processed, label))

        if i%1000 == 0: print(i)
        i += 1

    return new_set

img_size = (28,28)

train_set = torchvision.datasets.MNIST(
    root='./data/MNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_set = torchvision.datasets.MNIST(
    root='./data/MNIST'
    ,train=False
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

new_train_set = processDataSet(train_set)
new_test_set = processDataSet(test_set)

PATH_TRAIN = r'D:/DATABASE/TRAIN/'
PATH_TEST = r'D:/DATABASE/TEST/'

i = 0
for img, label in new_train_set:
    PATH_to_img = str(PATH_TRAIN+"img/"+str(i)+".jpg")
    PATH_to_label = str(PATH_TRAIN+str(i)+".txt")
    cv2.imwrite(PATH_to_img, img)
    with open(PATH_to_label, 'w') as f:
        f.write(PATH_to_img)
        f.write("\n"+str(label))
        f.close()
    if i%1000 == 0: print(i)
    i += 1 

i = 0
for img, label in new_test_set:
    PATH_to_img = str(PATH_TEST+"img/"+str(i)+".jpg")
    PATH_to_label = str(PATH_TEST+str(i)+".txt")
    cv2.imwrite(PATH_to_img, img)
    with open(PATH_to_label, 'w') as f:
        f.write(PATH_to_img)
        f.write("\n"+str(label))
        f.close()
    if i%1000 == 0: print(i)
    i += 1   