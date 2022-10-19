from torch.nn.modules import loss
from torch.optim.adam import Adam
from basic_unet_model import Unet  # basic_unet_model
from Mydataset import MyDataset  # self-defined Dataset
from util import ssim
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from math import log10
import os
import datetime
import time
import argparse


# Visualization using TensorBoardX
def tensorboardX_writer(train_writer, val_writer, time_stamp):
    data_path = os.path.join('/workspace/smbohann/Unettry/logs', time_stamp)
    log_dir = os.path.join(data_path, train_writer)
    train_writer = SummaryWriter(log_dir=log_dir)
    log_dir = os.path.join(data_path, val_writer)
    val_writer = SummaryWriter(log_dir=log_dir)
    return train_writer, val_writer


def evaluate_standards(loader, net, loss_function, optimizer, train_enable=None, device=None):
    criterion_mae = nn.L1Loss()
    sum_loss = 0.0
    sum_mae = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0

    for image, label in loader:
        # data normalization
        image = image / image.max()
        label = label / label.max()

        image = image.float()
        label = label.float()
        image, label = image.to(device), label.to(device)

        pred = net(image)

        loss = loss_function(pred, label)
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

    epoch_loss = sum_loss / len(loader)
    epoch_mae = sum_mae / len(loader)
    epoch_psnr = sum_psnr / len(loader)
    epoch_ssim = sum_ssim / len(loader)

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


# Training settings
def parser():
    parser = argparse.ArgumentParser(description='Please setting training parameters')
    parser.add_argument('--dataset', dest='data_path', default='/workspace/smbohann/Unettry/mydataset',
                        help='dataset path')
    # parser.add_argument('--dataset', dest='data_path', default='/home/WIN-UNI-DUE/sotiling/WS20/small_dataset', help='dataset path')
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int,
                        help='batch size for one iteration, default: 2')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='number of epochs, default: 10')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='value of learning rate, default: 0.001')
    parser.add_argument('--loss_function', dest='loss_function', default=nn.MSELoss(),
                        help='type of loss function, default: MSELoss')
    # parser.add_argument('--time_stamp', dest='time_stamp', default=None, type=str, help='The time_stamp of the model, you want to load.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    starttime = time.time()
    opt = parser()
    patient_list = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    train_val_list = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13]  # 10
    test_list = [14, 15]

    train_list = train_val_list[:8]
    val_list = train_val_list[8:]
    # random split will be tried later
    # train_list, val_list = torch.utils.data.random_split(train_val_list, [8,2])

    train_dataset = MyDataset(data_path=opt.data_path,
                              patientList=train_list,
                              repeat=1,
                              transform=transforms.Compose([transforms.ToTensor(),transforms.RandomAffine(degrees=(-15,15),translate=(0.1,0.1),shear=(10),scale=(0.88,1))]))

    val_dataset = MyDataset(data_path=opt.data_path,
                            patientList=val_list,
                            repeat=1,
                             transform=transforms.Compose([transforms.ToTensor(),transforms.RandomAffine(degrees=(-15,15),translate=(0.1,0.1),shear=(10),scale=(0.88,1))])))
    test_dataset = MyDataset(data_path=opt.data_path,
                             patientList=test_list,
                             repeat=1,
                              transform=transforms.Compose([transforms.ToTensor(),transforms.RandomAffine(degrees=(-15,15),translate=(0.1,0.1),shear=(10),scale=(0.88,1))]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # basic_unet_model
    net = Unet(1, 1)
    net.to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-8)

    # loading data
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=16)

    # timestamp
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_save_path = '/workspace/smbohann/Unettry/saved_models'
    model_name = os.path.join(model_save_path, 'best_model_{}.pth'.format(time_stamp))
    print(model_name)

    print('=' * 5, 'start of training', '=' * 5)

    for epoch in range(opt.epochs):
        train_loss, train_mae, train_psnr, train_ssim = train(train_loader, net, optimizer, opt.loss_function, device)
        val_loss, val_mae, val_psnr, val_ssim = val(val_loader, net, optimizer, opt.loss_function, device)
        min_loss = 1000
        print("origin")
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(net.state_dict(), model_name)

        print('epoch:{}'.format(epoch + 1))
        print('train_Loss/MSE: {:.6f} val_Loss/MSE: {:.6f}'.format(train_loss, val_loss))
        print('train_MAE: {:.6f} val_MAE: {:.6f}'.format(train_mae, val_mae))
        print('train_PSNR: {:.6f} val_PSNR: {:.6f}'.format(train_psnr, val_psnr))
        print('train_SSIM: {:.6f} val_SSIM: {:.6f}'.format(train_ssim, val_ssim))

        # TensorboardX Train and Validation Loss
        train_loss_writer, val_loss_writer = tensorboardX_writer('train_loss', 'val_loss', time_stamp)
        train_loss_writer.add_scalar('Loss/MSE', train_loss, epoch + 1)
        val_loss_writer.add_scalar('Loss/MSE', val_loss, epoch + 1)

        # TensorboardX Train and Validation MAE
        train_mae_writer, val_mae_writer = tensorboardX_writer('train_mae', 'val_mae', time_stamp)
        train_mae_writer.add_scalar('MAE', train_mae, epoch + 1)
        val_mae_writer.add_scalar('MAE', val_mae, epoch + 1)

        # TensorboardX Train and Validation PSNR
        train_psnr_writer, val_psnr_writer = tensorboardX_writer('train_psnr', 'val_psnr', time_stamp)
        train_psnr_writer.add_scalar('PSNR', train_psnr, epoch + 1)
        val_psnr_writer.add_scalar('PSNR', val_psnr, epoch + 1)

        # TensorboardX Train and Validation SSIM
        train_ssim_writer, val_ssim_writer = tensorboardX_writer('train_ssim', 'val_ssim', time_stamp)
        train_ssim_writer.add_scalar('SSIM', train_ssim, epoch + 1)
        val_ssim_writer.add_scalar('SSIM', val_ssim, epoch + 1)
    summary(net, input_size=(1,384,384))

    print('=' * 5, 'End of training ', '=' * 5)
    print('name of saved best model', model_name)
    print('=' * 5, 'Start of testing', '=' * 5)

    net.load_state_dict(torch.load(model_name, map_location=device))

    test_loss, test_mae, test_psnr, test_ssim = val(test_loader, net, optimizer, opt.loss_function, device)
    print('test_Loss/MSE: {:.6f} '.format(test_loss))
    print('test_MAE: {:.6f}'.format(test_mae))
    print('test_PSNR: {:.6f}'.format(test_psnr))
    print('test_SSIM: {:.6f}'.format(test_ssim))

    print('=' * 5, 'End of testing', '=' * 5)

    endtime = time.time()
    print(endtime - starttime)
