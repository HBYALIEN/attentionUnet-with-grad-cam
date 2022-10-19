from torch.nn.modules import loss
from basic_unet_model import Unet 
from Mydataset import MyDataset
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
import os
from util import ssim
from math import log10

def evaluate_standards(loader, net, loss_function, optimizer, train_enable=None, device=None):
    criterion_mae = nn.L1Loss()
    sum_loss = 0.0
    sum_mae = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    a = 0
    for image, label in loader:
        # data normalization
        
        #image = image / image.max()
        #label = label / label.max()
       
        image = image.float()
        label = label.float()
        image, label = image.to(device), label.to(device)

        pred = net(image)
        loss = loss_function(pred, label)
        print(loss)
        mae = criterion_mae(pred, label)
        psnr = 10 * log10(1 / loss.item())
        running_ssim = ssim(pred, label)

        if train_enable == 'True':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sum_loss += float(loss.item())
        sum_mae += float(mae.item())
        sum_psnr += float(psnr)
        sum_ssim += float(running_ssim)
        
    print(sum_loss)
    epoch_loss = sum_loss / len(loader)
    epoch_mae = sum_mae / len(loader)
    epoch_psnr = sum_psnr / len(loader)
    epoch_ssim = sum_ssim / len(loader)
    print(epoch_loss)
    return epoch_loss, epoch_mae, epoch_psnr, epoch_ssim


def train(train_loader, net, optimizer, loss_function, device=None):
    net = net.train()
    train_loss, train_mae, train_psnr, train_ssim = evaluate_standards(train_loader, net, loss_function, optimizer,
                                                                       train_enable='True', device=device)
    return train_loss, train_mae, train_psnr, train_ssim


def val(val_loader, net, optimizer, loss_function, device=None):
    net = net.eval()
    val_loss, val_mae, val_psnr, val_ssim = evaluate_standards(val_loader, net, loss_function, optimizer,
                                                               train_enable='False', device=device)
    return val_loss, val_mae, val_psnr, val_ssim

if __name__ == '__main__':

    patient_list = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    train_val_list = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13]  # 10
    data_path = '/workspace/smbohann/Unettry/mydataset'

    train_list = train_val_list[:9]
    val_list = train_val_list[9:]
    # random split will be tried later
    # train_list, val_list = torch.utils.data.random_split(train_val_list, [8,2])

    train_dataset = MyDataset(data_path,
                              train_list,
                              repeat=1,
                              transform=transforms.Compose([transforms.ToTensor()]))

    val_dataset = MyDataset(data_path,
                            val_list,
                            repeat=1,
                            transform=transforms.Compose([transforms.ToTensor()]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # basic_unet_model
    net = Unet(1, 1)
    net.to(device=device)

    # set training and validation process
    loss_function = nn.MSELoss()
    batch_size = 2
    epochs = 15
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    # loading data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    for epoch in range(epochs):
        train_loss, train_mae, train_psnr, train_ssim = train(train_loader, net, optimizer, loss_function, device)
        val_loss, val_mae, val_psnr, val_ssim = val(val_loader, net, optimizer, loss_function, device)
        best_loss = 0.0

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(net.state_dict(), 'best_model.pth')
        print('epoch:{}'.format(epoch + 1))

        print('train_Loss/MSE: {:.6f} val_Loss/MSE: {:.6f}'.format(train_loss, val_loss))
        print('train_MAE: {:.6f} val_MAE: {:.6f}'.format(train_mae, val_mae))
        print('train_PSNR: {:.6f} val_PSNR: {:.6f}'.format(train_psnr, val_psnr))
        print('train_SSIM: {:.6f} val_SSIM: {:.6f}'.format(train_ssim, val_ssim))
        print('=' * 20)
