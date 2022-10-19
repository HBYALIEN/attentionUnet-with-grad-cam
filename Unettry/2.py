from SimpleITK.SimpleITK import TranslationTransform
from basic_unet_model import Unet  # basic_unet_model
from attention_unet import AttU_Net
from datasetclass import MyDataset  # self-defined Dataset
from util import ssim_function
from torch.nn.modules import loss
from torch.optim.adam import Adam
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
from math import log10
import os
import datetime
import time
import argparse
import matplotlib.pyplot as plt
from PIL import Image


def parser():
    parser = argparse.ArgumentParser(description='Please setting training parameters')
    # parser.add_argument('--dataset', dest='data_path', default='/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl',help='dataset path')
    parser.add_argument('--dataset', dest='data_path', default='/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl', help='dataset path')
    # based on small dataset, 160 images
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int,
                        help='batch size for one iteration, default: 2')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='number of epochs, default: 10')
    parser.add_argument('--lr', dest='lr', default=0.0001, type=float, help='value of learning rate, default: 0.001')
    # ToDo: in future, loss_function may not be MSELoss()
    parser.add_argument('--loss_function', dest='loss_function', default=nn.MSELoss(),
                        help='type of loss function, default: MSELoss')
    # parser.add_argument('--time_stamp', dest='time_stamp', default=None, type=str, help='The time_stamp of the model, you want to load.')
    opt = parser.parse_args()
    args = vars(parser.parse_args())
    return opt

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))

        
        self.classifier = nn.Sequential(
            nn.Linear(512*64*18,64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64,num_classes))
        
    def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
            return x

if __name__ == '__main__':
    opt = parser()
    train_list = [1,2,5,6,7,9,10,11]
    val_list = [12,13]
    train_dataset = MyDataset(data_path=opt.data_path,
                             patientList =train_list,
                             repeat =1,
                             transform=transforms.Compose([transforms.ToTensor()]))
    val_dataset = MyDataset(data_path=opt.data_path,
                            patientList=val_list,
                            repeat=1,
                            transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=2)
    model.to(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
     
    no_epochs = 100
    train_loss = list()
    val_loss = list()
    best_val_loss = 1
    for epoch in range(no_epochs):
        total_train_loss = 0
        total_val_loss = 0

    model.train()
    for itr, (image, label) in enumerate(train_loader):
        image = image.to(device=device)
        label = label.to(device=device)
        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()*image.size(0)

        loss.backward()
        optimizer.step()
    total_train_loss = total_train_loss / len(train_loader.dataset)
    train_loss.append(total_train_loss)
    
    model.eval()
    total = 0
    for itr, (image, label) in enumerate(val_loader):
        image = image.to(device=device)
        label = label.to(device=device)
        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()*image.size(0)
        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
    
    accuracy = total / len(val_dataset)

    total_val_loss = total_val_loss / len(val_loader.dataset)
    val_loss.append(total_val_loss)
    
    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_val_loss))
        torch.save(model.state_dict(), "model.dth")