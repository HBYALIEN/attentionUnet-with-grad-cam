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
        slice_index=[300]

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
        return img_NAC.float()


    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len
        return data_len
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2))

        
        self.classifier = nn.Sequential(
            nn.Linear(512*64*18,64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64,num_classes))
        
    def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
            return x

import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.model_features = model.features
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model_features._modules.items():
            print(x.shape)
            
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        x = x.view(x.size(0), -1) 
        x = self.model.classifier(x)
        return outputs, x

class ModelOutputs():
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.5],
                                 std=[0.5])
    preprocessing = transforms.Compose([
        normalize
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cam = heatmap + np.float32(img_rgb)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = FeatureExtractor(self.model, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)
        print(1)
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        print(2)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        print(grads_val.shape)
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        print(target)
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output
def parser():
    parser = argparse.ArgumentParser(description='Please setting training parameters')
    # parser.add_argument('--dataset', dest='data_path', default='/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl',help='dataset path')
    parser.add_argument('--dataset', dest='data_path', default='dataset', help='dataset path')
    # based on small dataset, 160 images
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int,
                        help='batch size for one iteration, default: 2')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='number of epochs, default: 10')
    opt = parser.parse_args()
    args = vars(parser.parse_args(args=[]))
    return opt
import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    opt =parser()
    model1 = torch.load('model6.dth', map_location=torch.device('cpu'))
    model = CNN(num_classes=2)
    model.load_state_dict(model1)
    test_list=[15]
    train_dataset = MyDataset(data_path=opt.data_path,
                             patientList =train_list,
                             repeat =1,
                             transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    for itr, (image, label) in enumerate(train_loader):
        img = image.cuda()