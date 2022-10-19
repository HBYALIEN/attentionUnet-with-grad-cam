from SimpleITK.SimpleITK import TranslationTransform
from numpy.core.fromnumeric import mean

from attention_unet1 import att_unet


from Mydataset import MyDataset  # self-defined Dataset

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
from math import log10
import os
import datetime
import time
import argparse
import matplotlib.pyplot as plt

import SimpleITK as sitk


#
# öliuztredcfgvhuz6tgb0Training settings
def parser():
    parser = argparse.ArgumentParser(description='Please setting training parameters')
    parser.add_argument('--dataset', dest='data_path', default='/workspace/smbohann/Unettry/mydataset',
                        help='dataset path')
    # parser.add_argument('--dataset', dest='data_path', default='/home/WIN-UNI-DUE/sotiling/WS20/new_dataset', help='dataset path')
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int,
                        help='batch size for one iteration, default: 2')
    parser.add_argument('--epochs', dest='epochs', default=20, type=int, help='number of epochs, default: 10')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='value of learning rate, default: 0.001')
    parser.add_argument('--loss_function', dest='loss_function', default=nn.MSELoss(),
                        help='type of loss function, default: MSELoss')
    parser.add_argument('--num_workers', dest='num_workers', default=2, type=int, help='value of numbers')
    parser.add_argument('--train', dest='train', default=True, help='do training')
    parser.add_argument('--checkpoint_file', dest='checkpoint_file', default=None, help='input the path of the model')
    # ToDo: add default checkpoint_file
    opt = parser.parse_args()
    args = vars(parser.parse_args())
    print(args)
    return opt


MAX_C = 1


# data normalization
def MaxMinnormalization(img, device=None):
    normalized_img = img / MAX_C
    normalized_img = normalized_img.float()
    normalized_img = normalized_img.to(device)
    return normalized_img


# evaluation maxtrics with no normalized data
def matrics(img_AC, pred):
    mse = nn.MSELoss()(pred, img_AC)
    mae = nn.L1Loss()(pred, img_AC)
    L = pow(2, 18) - 1
    psnr = 10 * log10(pow(L, 2) / mse)  # ca.216373 is the biggest possible intensity of img_AC
    ssim = ssim_function(pred, img_AC, L=L)
    return mse, mae, psnr, ssim


# method for training and evaluation
def train_val(loader=None, model=None, loss_function=None, optimizer=None, train_enable=None, device=None):
    sum_loss = 0.0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0

    if train_enable == 'True':
        model = model.train()
    else:
        model = model.eval()  # default closing Dropout

    for img_NAC, img_AC, _ in loader:

        # do data normalization
        normalized_img_NAC = MaxMinnormalization(img_NAC, device=device)
        normalized_img_AC = MaxMinnormalization(img_AC, device=device)

        # input img_NAC into model
        pred = model(normalized_img_NAC)  # which means pred also normalized
        loss = loss_function(pred, normalized_img_AC)  # Loss is just MSE

        # for evaluation only use data without data normalization
        # rescaled_pred = pred * img_AC.max()
        rescaled_pred = pred * MAX_C

        # rescaled_img_AC = normalized_img_AC * img_NAC.max() # avoid the truncation error
        rescaled_img_AC = normalized_img_AC * MAX_C
        mse, mae, psnr, ssim = matrics(rescaled_img_AC, rescaled_pred)

        if train_enable == 'True':
            optimizer.zero_grad()
            loss.backward()  # back propagation
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


# method for testing
def test(loader=None, model=None, device=None):
    mse_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    model = model.eval()

    for img_NAC, img_AC, _ in loader:
        # data normalization
        normalized_img_NAC = MaxMinnormalization(img_NAC, device=device)
        normalized_img_AC = MaxMinnormalization(img_AC, device=device)

        # input img_NAC into model
        pred = model(normalized_img_NAC)

        # for evaluation only use data without data normalization
        rescaled_pred = pred * MAX_C
        rescaled_img_AC = normalized_img_AC * MAX_C  # avoid the truncation error
        mse, mae, psnr, ssim = matrics(rescaled_img_AC, rescaled_pred)

        mse_list.append(float(mse.item()))
        mae_list.append(float(mae.item()))
        psnr_list.append(float(psnr))
        ssim_list.append(float(ssim))

    epoch_mse = np.sum(mse_list) / len(loader)
    epoch_mae = np.sum(mae_list) / len(loader)
    epoch_psnr = np.sum(psnr_list) / len(loader)
    epoch_ssim = np.sum(ssim_list) / len(loader)

    std_mse = np.std(mse_list)
    std_mae = np.std(mae_list)
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)

    return epoch_mse, epoch_mae, epoch_psnr, epoch_ssim, std_mse, std_mae, std_psnr, std_ssim


# Visualization using TensorBoardX
def tensorboardX_writer(train_writer, val_writer, time_stamp, graph_name, train_value, val_value, epoch_num):
    data_path = os.path.join('/workspace/smbohann/Unettry/logs', time_stamp)
    log_dir = os.path.join(data_path, train_writer)
    train_writer = SummaryWriter(log_dir=log_dir)
    log_dir = os.path.join(data_path, val_writer)
    val_writer = SummaryWriter(log_dir=log_dir)

    train_writer.add_scalar(graph_name, train_value, epoch_num)
    val_writer.add_scalar(graph_name, val_value, epoch_num)


def main():
    opt = parser()

    # complete dataset with fixed split
    patient_list = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]  #
    train_list = [1, 2, 5, 6, 7, 9, 10, 11]
    val_list = [12, 13]
    test_list = [14, 15]

    # small dataset for prototype with fixed split
    # train_list = [1]
    # val_list = [12]
    # test_list = [15]

    # cross validation
    # kf = KFold(n_splits=5,shuffle=True) # or use False
    # for train_index, val_index in kf.split(train_val_list):

    #     train_list = np.array(train_val_list)[train_index]
    #     val_list = np.array(train_val_list)[val_index]

    #     print('train_list:',train_list)
    #     print('val_list:',val_list)

    # normal dataset
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

    # new dataset with gipl file are separared into slice gipl files.
    # train_dataset = new_MyDataset(data_path=opt.data_path,
    #                         patientList =train_list,
    #                         repeat=1,
    #                         transform=transforms.Compose([transforms.ToTensor()]))

    # val_dataset = new_MyDataset(data_path=opt.data_path,
    #                         patientList=val_list,
    #                         repeat=1,
    #                         transform=transforms.Compose([transforms.ToTensor()]))

    # test_dataset = new_MyDataset(data_path= opt.data_path,
    #                         patientList=test_list,
    #                         repeat=1,
    #                         transform=transforms.Compose([transforms.ToTensor()]))

    model = att_unet()
    # pool of models: unet(), att_unet(),resnet18_unet(), att_resnet18_unet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-8)

    # loading data
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=opt.num_workers)

    checkpoint_file = opt.checkpoint_file
    if opt.train:
        if checkpoint_file == None:
            # timestamp
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            dir_path = '/workspace/smbohann/Unettry/checkpoint'
            checkpoint_file = os.path.join(dir_path, 'best_model_{}.pth'.format(time_stamp))

        print('=' * 5, 'start of training', '=' * 5)

        for epoch in range(opt.epochs):
            train_loss, train_mse, train_mae, train_psnr, train_ssim = train_val(loader=train_loader,
                                                                                 model=model,
                                                                                 loss_function=opt.loss_function,
                                                                                 optimizer=optim.Adam(
                                                                                     model.parameters(), lr=opt.lr,
                                                                                     weight_decay=1e-8),
                                                                                 train_enable='True',
                                                                                 device=device)
            _, val_mse, val_mae, val_psnr, val_ssim = train_val(loader=val_loader,
                                                                model=model,
                                                                loss_function=opt.loss_function,
                                                                train_enable='False',
                                                                device=device)
            min_loss = 10000000

            if train_loss < min_loss:
                min_loss = train_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(checkpoint, checkpoint_file)
                print('checkpoint is saved in path', checkpoint_file)

            print('epoch:{}'.format(epoch + 1))
            print('train_Loss: {:.6f} '.format(train_loss))
            print('train_MSE: {:.6f} val_MSE: {:.6f}'.format(train_mse, val_mse))
            print('train_MAE: {:.6f} val_MAE: {:.6f}'.format(train_mae, val_mae))
            print('train_PSNR: {:.6f} val_PSNR: {:.6f}'.format(train_psnr, val_psnr))
            print('train_SSIM: {:.6f} val_SSIM: {:.6f}'.format(train_ssim, val_ssim))

            # TensorboardX
            # ToDo: fix problem with time_stamp
            # tensorboardX_writer('train_mse','val_mse',time_stamp,'MSE',train_mse,val_mse,epoch+1)
            # tensorboardX_writer('train_mae','val_mae',time_stamp,'MAE',train_mae,val_mae,epoch+1)
            # tensorboardX_writer('train_psnr','val_psnr',time_stamp,'PSNR',train_psnr,val_psnr,epoch+1)
            # tensorboardX_writer('train_ssim','val_ssim',time_stamp,'SSIM',train_ssim,val_ssim,epoch+1)

        print('=' * 5, 'End of training ', '=' * 5)
        print('name of saved best model', checkpoint_file)

    print('=' * 5, 'Start of testing', '=' * 5)
    if checkpoint_file == None:
        print('error! no model exists')
        return
    else:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint(['epoch'])

    test_mse, test_mae, test_psnr, test_ssim, std_mse, std_mae, std_psnr, std_ssim = test(loader=test_loader,
                                                                                          model=model,
                                                                                          device=device
                                                                                          )
    print('test_MSE ± STD: {:.6f} ±  {:.6f}'.format(test_mse, std_mse))
    print('test_MAE ± STD: {:.6f}  ± {:.6f}'.format(test_mae, std_mae))
    print('test_PSNR ± STD: {:.6f} ± {:.6f}'.format(test_psnr, std_psnr))
    print('test_SSIM ± STD: {:.6f} ± {:.6f}'.format(test_ssim, std_ssim))
    print('=' * 5, 'End of testing', '=' * 5)


if __name__ == '__main__':
    starttime = time.time()
    main()
    endtime = time.time()
    print('This run lasted', endtime - starttime, 'seconds')
