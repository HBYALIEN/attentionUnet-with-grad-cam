import os
import numpy as np
import SimpleITK as sitk
from SimpleITK import sitkNearestNeighbor
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch
device = "cuda"

def resize_sitk_2D(image_array, outputSize, interpolator=sitk.sitkLinear):
    #print("Shape of input slice:", image_array.shape)
    image = sitk.GetImageFromArray(image_array)
    inputSize = image.GetSize()
    inputSpacing = image.GetSpacing()
    outputSpacing = [1, 1]
    outputSpacing[0] = inputSpacing[0] * (inputSize[0] / outputSize[0])
    outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1])
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image1 = resampler.Execute(image)
    #print(1)
    resampled_arr = sitk.GetArrayFromImage(image1)
    #print(2)
    return resampled_arr


class MyDataset(Dataset):
    def __init__(self, patientList, transform):
        self.transform = transform
        self.data_list = []
        for patient_ix in patientList:
            # '/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient'+f'{patient_ix:02d}'
            p_path_NAC = '/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient' + f'{patient_ix:02d}' + '/' + 'Patient' + f'{patient_ix:02d}' + '_NAC.gipl'
            p_path_AC = '/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient' + f'{patient_ix:02d}' + '/' + 'Patient' + f'{patient_ix:02d}' + '_AC.gipl'
            sitk_vol = sitk.ReadImage(p_path_NAC)
            np_vol = sitk.GetArrayFromImage(sitk_vol)
            for i in range(np_vol.shape[0]):
                self.data_list.append([p_path_NAC, p_path_AC, i, np_vol.shape])

    def __getitem__(self, index):
        ls_item = self.data_list[index]
        p_path_NAC = ls_item[0]
        p_path_AC = ls_item[1]
        slice_index = ls_item[2]
        vol_dims = ls_item[3]

        sitk_vol = sitk.ReadImage(p_path_NAC)
        np_vol_NAC = sitk.GetArrayFromImage(sitk_vol)
        img_NAC = np_vol_NAC[slice_index, :, :]
        img_NAC = resize_sitk_2D(img_NAC, (384, 384))

        sitk_vol1 = sitk.ReadImage(p_path_AC)
        np_vol_AC = sitk.GetArrayFromImage(sitk_vol1)
        img_AC = np_vol_AC[slice_index, :, :]
        img_AC = resize_sitk_2D(img_AC, (384, 384))

        img_NAC = self.transform(img_NAC)
        img_AC = self.transform(img_AC)
        return img_NAC, img_AC

    def __len__(self):
        return len(self.data_list)


#train_val_patients_list = [1]
#train_val_dataset = MyDataset(train_val_patients_list, transform=transforms.Compose([transforms.ToTensor(),transforms.RandomAffine(degrees=(-30,30),translate=(0.1,0.1),scale=(0.88,0.88))]))
#train_val_loader = DataLoader(train_val_dataset, batch_size=2, shuffle=False, num_workers=1)

import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10



#or itr, (X_train, Y_train) in enumerate(train_val_loader):
    #X_train = X_train.type(torch.FloatTensor)
    #X_train, Y_train = X_train.to(device), Y_train.to(device)


from torch.utils.data.sampler import RandomSampler
from torch import optim
import torch.nn as nn
import torch
import numpy as np

from torch.utils.data import DataLoader
import torchvision.utils as vutils

def train(train_loader,net,optimizer,criterion_loss,criterion_acc,device):
    train_loss = 0.0
    train_acc = 0.0
    net = net.train() # start batch normalization
    for itr, (X_train, Y_train) in enumerate(train_loader):
            X_train = X_train.type(torch.FloatTensor)
            X_train = X_train.float()
            Y_train = Y_train.float()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            padded_img = X_train
            padded_label = Y_train
            pred = net(padded_img)
            loss = criterion_loss(pred,padded_label)
            acc = criterion_acc(pred,padded_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(6)

            train_loss += float(loss.item())
            train_acc += float(acc.item())

    epoch_loss = train_loss/len(train_loader)

    epoch_acc = train_acc/len(train_loader)

    return epoch_loss,epoch_acc

def val(val_loader,net,criterion_loss,criterion_acc,device):
    val_loss = 0.0
    val_acc = 0.0
    net = net.eval() # stop batch normalization
    for itr, (X_train, Y_train) in enumerate(val_loader):
            X_train = X_train.float()
            Y_train = Y_train.float()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            padded_img = X_train
            padded_label = Y_train
            pred = net(padded_img) # tensor
            loss = criterion_loss(pred,padded_label)
            acc = criterion_acc(pred,padded_label)
            val_loss += float(loss.item())
            val_acc += float(acc.item())

    epoch_loss = val_loss/len(val_loader)
    epoch_acc = val_acc/len(val_loader)
    return epoch_loss,epoch_acc



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(1,1) 
    net.to(device=device)

    criterion_loss = nn.MSELoss()
    criterion_acc = nn.L1Loss()
    batch_size = 2
    epochs = 10
    lr = 0.001
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9) 
    train_val_list = [1,2,5,6,7,9,10,11,12,13] # 10
    
    
    train_list = train_val_list[:9]
    val_list = train_val_list[9:]
    train_data= MyDataset(train_list,transform=transforms.ToTensor())
    val_data= MyDataset(val_list,transform=transforms.ToTensor())

    train_loader = DataLoader(train_data,
                        shuffle=RandomSampler(train_data), 
                        batch_size=batch_size) 

    val_loader = DataLoader(val_data,
                        shuffle=False, 
                        batch_size=batch_size) 
    for epoch in range(epochs):
        print(1)
        train_loss,train_acc = train(train_loader,net,optimizer,criterion_loss,criterion_acc,device)
        print(2)
        val_loss,val_acc = val(val_loader,net,criterion_loss,criterion_acc,device)
        best_loss = 0.0
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(net.state_dict(),'best_model.pth')

        print('epoch: {} train_loss: {:.3f} val_loss: {:.3f}'.format(epoch+1, train_loss, val_loss))
        print('epoch: {} train_acc: {:.3f} val_acc: {:.3f}'.format(epoch+1, train_acc, val_acc))

