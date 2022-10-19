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
    print("Shape of input slice:", image_array.shape)
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
    print(1)
    resampled_arr = sitk.GetArrayFromImage(image1)
    print(2)
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


train_val_patients_list = [1]
train_val_dataset = MyDataset(train_val_patients_list, transform=transforms.Compose([transforms.ToTensor(),transforms.RandomAffine(degrees=(-30,30),translate=(0.1,0.1),scale=(1.22,1.22))]))
train_val_loader = DataLoader(train_val_dataset, batch_size=2, shuffle=False, num_workers=1)
for itr, (X_train, Y_train) in enumerate(train_val_loader):
    X_train = X_train.type(torch.FloatTensor)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
