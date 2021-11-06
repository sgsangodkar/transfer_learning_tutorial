#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 07:22:26 2021

@author: sagar
"""

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


""" STEP 1: Dataloading Functions """
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

"""
Subclass ImageFolder inherits base class 'Dataset'.
It overwrides __getitem__ and __len__ methods.
"""

train_dataset = ImageFolder('data/seg_train/seg_train', transform=None)
test_dataset = ImageFolder('data/seg_test/seg_test', transform=None)

"""
class TLDataset(Dataset):
    def __init__(self, *args):
        # All initialisations
             
    def __len__(self):
        # Return length of the dataset

    def __getitem__(self, idx):
        # Return image and label corresponding to idx
"""
import cv2
class TLDataset(Dataset):
    def __init__(self, path_file, transform=None):
        # Define the instance variables
        self.transform = transform
        with open(path_file, 'r') as f:
            self.paths = f.readlines()
             
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label, img_path = self.paths[idx].strip().split(',')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)


        return img, int(label)

# Image.fromarray(img)
"""
Lets define path_file and transform for train and test set
"""
from torchvision import transforms

trainset_configs = dict(
    path_file = 'trainpaths.txt',

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((250,250)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
)


testset_configs = dict(
    path_file = 'testpaths.txt',

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
)

train_dataset = TLDataset(
        path_file = trainset_configs['path_file'],
        transform = trainset_configs['transform']
    )


test_dataset = TLDataset(
        path_file = testset_configs['path_file'],
        transform = testset_configs['transform']
    )

"""
Lets define the dataloaders.
"""

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 2,
    pin_memory = torch.cuda.is_available(),
    drop_last = True
)

test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size = 2,
    shuffle = False,
    num_workers = 2,
    pin_memory = torch.cuda.is_available(),
    drop_last = True
)

iterator = iter(train_dataloader)
images, labels = next(iterator)

"""
STEP 2: Defining the model and initialising with pretrained weights.
"""
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512,6)
model = model.to(device)

"""
STEP 3: Loss functions and optimiser
"""
import torch.optim as optim
from torch.optim import lr_scheduler

loss_function = nn.CrossEntropyLoss()

optimiser = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.fc.parameters(), 'lr': 0.01}   
    ], lr=0.0001, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 10 epochs
scheduler = lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.1)

"""
Step 4: Put all together. That is, define training loop
"""
dataloaders = dict(
        train = train_dataloader,
        test = test_dataloader
        )

from train import train_model
trained_model = train_model(model, 
                            dataloaders, 
                            loss_function, 
                            optimiser, 
                            scheduler, 
                            num_epochs=10
                )
