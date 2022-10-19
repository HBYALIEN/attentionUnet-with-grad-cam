import os
import numpy as np
import SimpleITK as sitk
from SimpleITK import sitkNearestNeighbor
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def resize_sitk_2D(image_array, outputSize, interpolator=sitk.sitkLinear):
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
    resampled_arr = sitk.GetArrayFromImage(image1)
    return resampled_arr

class MyDataset(Dataset):
    def __init__(self,patientList,transform):
        self.transform = transform
        self.data_list = []
        for patient_ix in patientList:
            #'/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient'+f'{patient_ix:02d}'
            p_path_NAC ='/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient'+f'{patient_ix:02d}' + '/' +'Patient' + f'{patient_ix:02d}' + '_NAC.gipl'
            p_path_AC ='/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient'+f'{patient_ix:02d}' + '/' +'Patient' + f'{patient_ix:02d}' + '_AC.gipl'
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
        img_NAC = np_vol_NAC[slice_index, : , : ]
        img_NAC = resize_sitk_2D(img_NAC, (384, 384))

        


        #img_NAC = Image.fromarray(img_NAC, mode="L")
        sitk_vol1 = sitk.ReadImage(p_path_AC)
        np_vol_AC = sitk.GetArrayFromImage(sitk_vol1)
        img_AC = np_vol_AC[slice_index, : , : ]
        img_AC = resize_sitk_2D(img_AC, (384, 384))


        #img_AC = np.expand_dims(img_AC, axis=0)
        




        #img_NAC = Image.fromarray(img_NAC, mode="L")
        
        
        img_NAC = self.transform(img_NAC)
        img_AC = self.transform(img_AC)
        return img_NAC, img_AC
    def __len__(self):
        return len(self.data_list)
train_val_patients_list = [1,2]
train_val_dataset = MyDataset(train_val_patients_list,transform=transforms.ToTensor())
train_val_loader = DataLoader(train_val_dataset,batch_size=2, shuffle=False, num_workers= 16)
import torch
import torch.nn as nn
if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm
import numpy.ma as ma
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, padding=500):
        super(UNet, self).__init__()

        # This parameter controls how far the UNet blocks grow as you go down 
        # the contracting path
        features = init_features

        # Below, we set up our layers
        self.encoder1 = self.unet_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.unet_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.unet_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.unet_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.unet_block(features * 8, features * 16, name="bottleneck")

        # Note the transposed convolutions here. These are the operations that perform
        # the upsampling. This is a blog post that explains them nicely:
        # https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self.unet_block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self.unet_block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self.unet_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self.unet_block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
        self.softmax = nn.Softmax(dim = 1)

    # This method runs the model on a data vector. Note that this particular
    # implementation is performing 2D convolutions, therefore it is 
    # set up to deal with 2D/2.5D approaches. If you want to try out the 3D convolutions
    # from the previous exercise, you will need to modify the initialization code
    def forward(self, x):
        # Contracting/downsampling path. Each encoder here is a set of 2x convolutional layers
        # with batch normalization, followed by activation function and max pooling
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # This is the bottom-most 1-1 layer.
        # In the original paper, a dropout layer is suggested here, but
        # we won't use it here since our dataset is tiny and we basically want 
        # to overfit to it
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Expanding path. Note how output of each layer is concatenated with the downsampling block
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out_conv = self.conv(dec1)
        
        return self.softmax(out_conv)



    # This method executes the "U-net block"
    def unet_block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
unet = UNet(1, 1) 
unet.float()
unet.to(device)
loss = nn.L1Loss()
optimizer = optim.Adam(unet.parameters(), lr=0.001)
optimizer.zero_grad()

unet.train()
for epoch in range(0,10):
    for itr,(X_train, Y_train) in enumerate(train_val_loader):
        print(X_train.shape)
        
        #X_train=  np.pad(X_train, [(0,0),(0,0),(20,20),(20, 20)], mode='constant')

        #Y_train = np.pad(Y_train, [(0,0),(0,0),(20,20),(20, 20)], mode='constant')
        #X_train =X_train/X_train.max() #basic normaization, may have loss
        #Y_train = Y_train/Y_train.max()

        #X_train = torch.from_numpy(X_train)#.permute(2, 0, 1)
        #Y_train = torch.from_numpy(Y_train)
        #Y_train = Y_train.float()
        #X_train, Y_train = X_train.to(device), Y_train.to(device)


        X_train = X_train.type(torch.FloatTensor)
        Y_train = Y_train.type(torch.FloatTensor)
        X_train, Y_train= X_train.to(device), Y_train.to(device)
        
        optimizer.zero_grad()

        pred = unet(Y_train)
        
        #print(torch.argmax(Y_train, )
  
        l = loss(pred,Y_train)
        l.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, training loss: {l}")
        