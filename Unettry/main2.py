from SimpleITK.SimpleITK import TranslationTransform
from basic_unet_model import Unet  # basic_unet_model
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
from torch.utils.data.sampler import RandomSampler
from math import log10
import os
import datetime
import time
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def tensorboardX_writer(train_writer, val_writer, time_stamp, graph_name, train_value, val_value, epoch_num):
    data_path = os.path.join('/workspace/smbohann/Unettry/logs', time_stamp)
    log_dir = os.path.join(data_path, train_writer)
    train_writer = SummaryWriter(log_dir=log_dir)
    log_dir = os.path.join(data_path, val_writer)
    val_writer = SummaryWriter(log_dir=log_dir)

    train_writer.add_scalar(graph_name, train_value, epoch_num)
    val_writer.add_scalar(graph_name, val_value, epoch_num)

    # return train_writer,val_writer


# maxtrix for evaluation
def matrices(reference_image, pred):
    mse = nn.MSELoss()(pred, reference_image)  # reduction = mean
    mae = nn.L1Loss()(pred, reference_image)
    psnr = 20 * log10(pred.max() / mse.item())  # MAXI of pred are not 1
    ssim = ssim_function(pred, reference_image)

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
        net = net.eval()  # default closing Dropout

    for image, reference_image, _ in loader:
        # data normalization
        image = image / image.max()
        reference_image = reference_image / reference_image.max()
        image = image.float()
        reference_image = reference_image.float()
        image, reference_image = image.to(device), reference_image.to(device)

        pred = net(image)

        loss = loss_function(pred, reference_image)
        mse, mae, psnr, ssim = matrices(reference_image, pred)

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


def test(loader, net, device=None, time_stamp=None):
    net = net.eval()

    mse_list = []
    #min_mse = 100.0
    mae_list = []
    psnr_list = []
    ssim_list = []

    for img_NAC, img_AC, name_img_AC in test_loader:
        # data normalization
        img_NAC = img_NAC / img_NAC.max()
        img_AC = img_AC / img_AC.max()
        img_NAC = img_NAC.float()
        img_AC = img_AC.float()
        img_NAC = img_NAC.to(device)
        img_AC = img_AC.to(device)


        pred = net(img_NAC)
        mse, mae, psnr, ssim = matrices(img_AC, pred)

        name_img_AC = "".join(name_img_AC)
        if name_img_AC == '/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl/Patient15/Patient15_AC.gipl145':
            chosen_img_NAC = img_NAC
            chosen_img_AC = img_AC
            chosen_pred = pred
            difference_image = chosen_pred - chosen_img_AC
            print('difference image comes from', name_img_AC)

            chosen_img_NAC = chosen_img_NAC.cpu().detach().numpy()
            chosen_img_AC = chosen_img_AC.cpu().detach().numpy()
            chosen_pred = chosen_pred.cpu().detach().numpy()
            difference_image = difference_image.cpu().detach().numpy()

            plt.figure(figsize=(20, 12))
            plt.subplot(1, 4, 1)
            plt.imshow(chosen_img_NAC[0, 0, 20:-20, 20:-20], cmap='gray')
            plt.title('img_NAC')

            plt.subplot(1, 4, 2)
            plt.imshow(chosen_img_AC[0, 0, 20:-20, 20:-20], cmap='gray')
            plt.title('img_AC')

            plt.subplot(1, 4, 3)
            plt.imshow(chosen_pred[0, 0, 20:-20, 20:-20], cmap='gray')
            plt.title('Prediction')

            plt.subplot(1, 4, 4)
            plt.imshow(difference_image[0, 0, 20:-20, 20:-20], cmap='rainbow')
            plt.title('difference_image')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.clim(0, 1)
            plt.show()

            dir_path = os.path.join('/workspace/smbohann/Unettry/diff')
            difference_image_path = os.path.join(dir_path, str(time_stamp) + 'difference.tif')
            plt.savefig(difference_image_path, bbox_inches="tight")
            print('difference_image is saved with following path')
            print(difference_image_path)

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


# Training settings
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
    print(args)
    return opt


if __name__ == '__main__':
    starttime = time.time()
    opt = parser()
    patient_list = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    train_list = [1, 2, 5, 6]  # 10
    val_list = [12]
    test_list = [15]


    # random split will be tried later
    # train_list, val_list = torch.utils.data.random_split(train_val_list, [8,2])

    # train without data augmentation
    #train_dataset = MyDataset(data_path=opt.data_path,
    #                          patientList=train_list,
    #                          repeat=1,
    #                          transform=transforms.Compose([transforms.ToTensor()]))

    #train with data augmentation
    train_dataset = MyDataset(data_path=opt.data_path,
                             patientList =train_list,
                             repeat =1,
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
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=2)

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

    test_mse, test_mae, test_psnr, test_ssim, std_mse, std_mae, std_psnr, std_ssim = test(test_loader, net=net,
                                                                                          device=device,  time_stamp=time_stamp)
    print('test_MSE: {:.6f} test_MSE_STD: {:.6f}'.format(test_mse, std_mse))
    print('test_MAE: {:.6f}, test_MAE_STD: {:.6f}'.format(test_mae, std_mae))
    print('test_PSNR: {:.6f}, test_PSNR_STD: {:.6f}'.format(test_psnr, std_psnr))
    print('test_SSIM: {:.6f}, test_SSIM_STD: {:.6f}'.format(test_ssim, std_ssim))

    print('=' * 5, 'End of testing', '=' * 5)
    #print('difference images with best test mse value are saved!')

    endtime = time.time()
    print(endtime - starttime)
