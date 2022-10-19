import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import pandas as pd

def default_loader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset):
    def __init__(self, csv, transform=None, target_transform=None, loader=default_loader):
        df = pd.read_table(csv,sep=',',header='infer')
        imgs = []
        for i in range(len(df)):
            imgs.append((df["input_path"][i],df["input_label"][i],df["output_path"][i],df["output_label"][i]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        #print(imgs)

    def __getitem__(self, index):
        fn_input, label_input,fn_output,label_output = self.imgs[index]
        input_img = self.loader(fn_input)
        output_img = self.loader(fn_output)
        if self.transform is not None:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)
        return input_img.cuda(),output_img.cuda()

    def __len__(self):
        return len(self.imgs)
train_data=MyDataset(csv='/workspace/smbohann/Unettry/111.csv', transform=transforms.ToTensor())
unet_trainset, unet_testset = torch.utils.data.random_split(train_data, [int(1/1000 * len(train_data)), len(train_data)-int(1/1000 * len(train_data))])
train_dataloader = torch.utils.data.DataLoader(unet_trainset, batch_size=64, shuffle=True)

import torch
import torch.nn as nn
if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=64):
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
        #self.encoder4 = self.unet_block(features * 4, features * 8, name="enc4")
        #self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.unet_block(features * 4, features * 8, name="bottleneck")
       
        # Note the transposed convolutions here. These are the operations that perform
        # the upsampling. This is a blog post that explains them nicely:
        # https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0
        #self.upconv4 = nn.ConvTranspose2d(
            #features * 8, features * 8, kernel_size=2, stride=2
       # )
        #self.decoder4 = self.unet_block((features * 8) * 2, features * 8, name="dec4")
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
        #enc4 = self.encoder4(self.pool3(enc3))

        # This is the bottom-most 1-1 layer.
        # In the original paper, a dropout layer is suggested here, but
        # we won't use it here since our dataset is tiny and we basically want 
        # to overfit to it
        #print(enc3.size())
        bottleneck = self.bottleneck(self.pool3(enc3))
        #print(bottleneck.size())
        # Expanding path. Note how output of each layer is concatenated with the downsampling block
        dec3 = self.upconv3(bottleneck)
        #print(dec3.size())
        #dec3 = torch.cat((dec3, enc3), dim=1)
        #dec4 = self.decoder4(dec4)
        #dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        #print(dec3.size())
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

# Move all trainable parameters to the device
unet.to(device)

# We will use Cross Entropy loss function for this one - we are performing per-voxel classification task, 
# so it should do ok.
# Later in the lesson we will discuss what are some of the other options for measuring medical image 
# segmentation performance.

loss = torch.nn.CrossEntropyLoss()

# You can play with learning rate later to see what yields best results
optimizer = optim.Adam(unet.parameters(), lr=0.001)
optimizer.zero_grad()

unet.train()
for epoch in range(0,30):
    for itr, (input_img, output_img) in enumerate(train_dataloader):
        #print(input_img.size())
        #print(output_img.size())
        optimizer.zero_grad()
        pred = unet(input_img)
        #target = target.squeeze(1)
        output_img = torch.argmax(output_img, dim=1)
        #print(pred.size())
        #print(output_img.size())
        l = loss(pred, output_img)
        
        l.backward()
       
        #loss = criterion(pred, output_img.unsqueeze(0))
        optimizer.step()
        print(f"Epoch: {epoch}, training loss: {l}")