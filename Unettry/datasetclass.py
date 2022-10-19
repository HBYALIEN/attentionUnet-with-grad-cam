import os
import numpy as np
import SimpleITK as sitk
from SimpleITK import sitkNearestNeighbor
from torch.utils.data import Dataset
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
    def __init__(self, data_path, patientList=None, repeat=1, transform=None):
        self.transform = transform
        self.data_list = []
        self.repeat = repeat

        for p in patientList:
            p_path_NAC = self.get_patient_path_NAC(data_path, p)
            p_path_AC = self.get_patient_path_AC(data_path, p)

            # to get the num of slices of each gipl
            sitk_vol = sitk.ReadImage(p_path_NAC)
            np_vol = sitk.GetArrayFromImage(sitk_vol)
            
        
            for i in range(np_vol.shape[0]):
                self.data_list.append([p_path_NAC, p_path_AC, i, np_vol.shape])
                # .... whatever you want to add, e.g. data augmentation type...)

        self.len = len(self.data_list)

    def get_patient_path_AC(self, data_path, patient_id):
        patient_path = os.path.join(data_path, "Patient%02d" % patient_id, "Patient%02d_AC.gipl" % patient_id)
        return patient_path

    def get_patient_path_NAC(self, data_path, patient_id):
        patient_path = os.path.join(data_path, "Patient%02d" % patient_id, "Patient%02d_NAC.gipl" % patient_id)
        return patient_path

    def __getitem__(self, index):
        ls_item = self.data_list[index]
        p_path_NAC = ls_item[0]
        p_path_AC = ls_item[1]
        slice_index = ls_item[2]
        list=[]
        #sitk_gassuian = sitk.SmoothingRecursiveGaussianImageFilter()
        #sitk_gassuian.SetSigma(1)
        #sitk_gassuian.NormalizeAcrossScaleOff()


        sitk_vol = sitk.ReadImage(p_path_NAC)
        #sitk_vol = sitk_gassuian.Execute(sitk_vol)
        np_vol_NAC = sitk.GetArrayFromImage(sitk_vol)
        img_NAC = np_vol_NAC[slice_index, :, :]
        img_NAC = resize_sitk_2D(img_NAC, (384, 384))

        sitk_vol1 = sitk.ReadImage(p_path_AC)
        #sitk_vol1 = sitk_gassuian.Execute(sitk_vol1)
        np_vol_AC = sitk.GetArrayFromImage(sitk_vol1)
        img_AC = np_vol_AC[slice_index, :, :]
        img_AC = resize_sitk_2D(img_AC, (384, 384))

        img_NAC = self.transform(img_NAC)
        img_AC = self.transform(img_AC)
        list.append((img_NAC.float(),0))
        list.append((img_AC.float(),1))
        for image, label in list:
            a= image
            b= label
        return a,b


    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len
        return data_len