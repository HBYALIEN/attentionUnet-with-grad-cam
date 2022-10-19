from torch.nn.modules import loss
from torch.optim.adam import Adam
from basic_unet_model import Unet  # basic_unet_model
from Mydataset import MyDataset  # self-defined Dataset
from util import ssim_function
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
import SimpleITK as sitk
from PIL import Image


# Visualization using TensorBoardX
def tensorboardX_writer(train_writer, val_writer, time_stamp, graph_name, train_value, val_value, epoch_num):
    data_path = os.path.join('/workspace/smbohann/Unettry/logs', time_stamp)
    log_dir = os.path.join(data_path, train_writer)
    train_writer = SummaryWriter(log_dir=log_dir)
    log_dir = os.path.join(data_path, val_writer)
    val_writer = SummaryWriter(log_dir=log_dir)

    train_writer.add_scalar(graph_name, train_value, epoch_num)
    val_writer.add_scalar(graph_name, val_value, epoch_num)




# maxtrix for evaluation
def matrices(label, pred):
    mse = nn.MSELoss()(pred, label)
    mae = nn.L1Loss()(pred, label)
    psnr = 10 * log10(1 / mse.item())
    ssim = ssim_function(pred, label)

    return mse, mae, psnr, ssim


def train_val(loader, net, loss_function, optimizer, train_enable=None, device=None):
    sum_loss = 0.0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    if train_enable == 'True':
        net = net.train()
    else:
        net = net.eval()

    for image, label in loader:
        # data normalization
        image = image / image.max()
        label = label / label.max()
        image = image.float()
        label = label.float()
        image, label = image.to(device), label.to(device)

        pred = net(image)

        loss = loss_function(pred, label)
        mse, mae, psnr, ssim = matrices(label, pred)

        if train_enable == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sum_loss += float(loss.item())
        sum_mse += float(mse.item())
        sum_mae += float(mae.item())
        sum_psnr += float(psnr)
        sum_ssim += float(ssim)

    epoch_loss = sum_loss / len(loader)
    epoch_mse = sum_mse / len(loader)
    epoch_mae = sum_mae / len(loader)
    epoch_psnr = sum_psnr / len(loader)
    epoch_ssim = sum_ssim / len(loader)

    return epoch_loss, epoch_mse, epoch_mae, epoch_psnr, epoch_ssim


def test(loader, net, device=None, save=False):
    net = net.eval()
    sum_mse = 0.0
    max_mse = 0.0
    min_mse = 100.0
    sum_mae = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    net = net.eval()
    worst_image = None
    worst_label = None
    worst_pred = None
    best_image = None
    best_label = None
    best_pred = None

    for image, label in test_loader:
        # data normalization
        image = image / image.max()
        label = label / label.max()
        image = image.float()
        label = label.float()
        image, label = image.to(device), label.to(device)

        pred = net(image)
        mse, mae, psnr, ssim = matrices(label, pred)

        if mse.item() > max_mse:
            max_mse = mse.item()
            worst_image = image
            worst_label = label
            worst_pred = pred
        if mse.item() < min_mse:
            min_mse = mse.item()
            best_image = image
            best_label = label
            best_pred = pred

        sum_mse += float(mse.item())
        sum_mae += float(mae.item())
        sum_psnr += float(psnr)
        sum_ssim += float(ssim)

    epoch_mse = sum_mse / len(loader)
    epoch_mae = sum_mae / len(loader)
    epoch_psnr = sum_psnr / len(loader)
    epoch_ssim = sum_ssim / len(loader)

    # save those images, calculate difference images
    best_image = best_image.cpu().detach().numpy()
    best_label = best_label.cpu().detach().numpy()
    best_pred = best_pred.cpu().detach().numpy()
    best_difference_image = best_pred - best_label

    worst_image = worst_image.cpu().detach().numpy()
    worst_label = worst_label.cpu().detach().numpy()
    worst_pred = worst_pred.cpu().detach().numpy()
    worst_difference_image = worst_pred - worst_label

    best_compare = np.zeros([344, 344 * 4])
    best_compare[:, 0:344] = best_image[0, 0, 20:-20, 20:-20]
    best_compare[:, 344:344 * 2] = best_label[0, 0, 20:-20, 20:-20]
    best_compare[:, 344 * 2:344 * 3] = best_pred[0, 0, 20:-20, 20:-20]
    best_compare[:, 344 * 3:344 * 4] = best_difference_image[0, 0, 20:-20, 20:-20]

    worst_compare = np.zeros([344, 344 * 4])
    worst_compare[:, 0:344] = worst_image[0, 0, 20:-20, 20:-20]
    worst_compare[:, 344:344 * 2] = worst_label[0, 0, 20:-20, 20:-20]
    worst_compare[:, 344 * 2:344 * 3] = worst_pred[0, 0, 20:-20, 20:-20]
    worst_compare[:, 344 * 3:344 * 4] = worst_difference_image[0, 0, 20:-20, 20:-20]

    plt.subplot(211)
    plt.imshow(best_compare)
    plt.title('image/label/pred/difference_image with best mse')
    plt.subplot(212)
    plt.imshow(worst_compare)
    plt.title('image/label/pred/difference_image with worst mse')
    plt.colorbar()
    plt.show()

    save_compare_path = '/workspace/smbohann/Unettry/diff'
    Image.fromarray(best_compare).save(os.path.join(save_compare_path, 'best_compare.tif'))
    Image.fromarray(worst_compare).save(os.path.join(save_compare_path, 'worst_compare.tif'))

    return epoch_mse, epoch_mae, epoch_psnr, epoch_ssim


# Training settings
def parser():
    parser = argparse.ArgumentParser(description='Please setting training parameters')
    # parser.add_argument('--dataset', dest='data_path', default='/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl', help='dataset path')
    parser.add_argument('--dataset', dest='data_path', default='/workspace/smbohann/Unettry/my1',
                        help='dataset path')
    # based on small dataset, 160 images
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int,
                        help='batch size for one iteration, default: 2')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='number of epochs, default: 10')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='value of learning rate, default: 0.001')
    # ToDo: in future, loss_function may not be MSELoss()
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
                              transform=transforms.Compose([transforms.ToTensor()]))

    val_dataset = MyDataset(data_path=opt.data_path,
                            patientList=val_list,
                            repeat=1,
                            transform=transforms.Compose([transforms.ToTensor()]))

    test_dataset = MyDataset(data_path=opt.data_path,
                             patientList=test_list,
                             repeat=1,
                             transform=transforms.Compose([transforms.ToTensor()]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # basic_unet_model
    net = Unet(1,1)

    # Attention_basic_unet_model
    # net = AttUnet(1,1)

    # transfer learning: resnet18_unet_model
    # net = resnet18_unet()

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

    print('=' * 5, 'start of training', '=' * 5)

    for epoch in range(opt.epochs):
        train_loss, train_mse, train_mae, train_psnr, train_ssim = train_val(train_loader, net, opt.loss_function,
                                                                             optimizer, train_enable=True,
                                                                             device=device)
        _, val_mse, val_mae, val_psnr, val_ssim = train_val(val_loader, net, opt.loss_function, optimizer,
                                                            train_enable=False, device=device)
        min_loss = 1000

        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(net.state_dict(), model_name)

        print('epoch:{}'.format(epoch + 1))
        print('train_Loss: {:.6f} '.format(train_loss))
        print('train_MSE: {:.6f} val_MSE: {:.6f}'.format(train_mse, val_mse))
        print('train_MAE: {:.6f} val_MAE: {:.6f}'.format(train_mae, val_mae))
        print('train_PSNR: {:.6f} val_PSNR: {:.6f}'.format(train_psnr, val_psnr))
        print('train_SSIM: {:.6f} val_SSIM: {:.6f}'.format(train_ssim, val_ssim))

        # TensorboardX
        tensorboardX_writer('train_mse', 'val_mse', time_stamp, 'MSE', train_mse, val_mse, epoch + 1)
        tensorboardX_writer('train_mae', 'val_mae', time_stamp, 'MAE', train_mae, val_mae, epoch + 1)
        tensorboardX_writer('train_psnr', 'val_psnr', time_stamp, 'PSNR', train_psnr, val_psnr, epoch + 1)
        tensorboardX_writer('train_ssim', 'val_ssim', time_stamp, 'SSIM', train_ssim, val_ssim, epoch + 1)

    print('=' * 5, 'End of training ', '=' * 5)
    print('name of saved best model', model_name)

    print('=' * 5, 'Start of testing', '=' * 5)
    net.load_state_dict(torch.load(model_name, map_location=device))

    test_mse, test_mae, test_psnr, test_ssim = test(test_loader, net=net, device=device, save=True)
    print('test_MSE: {:.6f}'.format(test_mse))
    print('test_MAE: {:.6f}'.format(test_mae))
    print('test_PSNR: {:.6f}'.format(test_psnr))
    print('test_SSIM: {:.6f}'.format(test_ssim))

    print('=' * 5, 'End of testing', '=' * 5)
    print('difference images with best and worse mse value are saved!')

    endtime = time.time()
    print(endtime - starttime)
