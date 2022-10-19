import torch
from torch import nn
from torchsummary import summary


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # for memeory burnout
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class att_unet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(att_unet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=1)

        # self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x1_p = self.Maxpool(x1)
        x2 = self.Conv2(x1_p)

        x2_p = self.Maxpool(x2)
        x3 = self.Conv3(x2_p)

        x3_p = self.Maxpool(x3)
        x4 = self.Conv4(x3_p)

        x4_p = self.Maxpool(x4)
        x5 = self.Conv5(x4_p)  # feature vectors

        # decoding + concat path
        d5 = self.Up5(x5)
        x4_att = self.Att5(d5, x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3_att = self.Att4(d4, x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(d3, x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(d2, x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # d1 = self.Conv_1x1(d2)

        return d2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model(model,model_path,device):
    model.to(device=device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


import cv2
import os
import SimpleITK as sitk
from torch.optim.adam import Adam
from torchvision import transforms
import torch.nn as nn
import datetime
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

def load_data(volume_path,slice_id):
        volume = sitk.ReadImage(volume_path)
        img = volume[:,:,slice_id]
        np_img = sitk.GetArrayFromImage(img)
        resize_img = resize_sitk_2D(np_img, (384, 384))
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_img = transform(resize_img)
        tensor_img = tensor_img.unsqueeze(0)
        return tensor_img
def img_transform(img,device):
    img = img.float()
    img = img.to(device=device)
    return img

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.model_features = model.Conv5
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model_features._modules.items():
            x = tensor_img_NAC
            x = x.float()
            x = x.to(device=device)
            
            x1 = model.Conv1(x)
            x1_p = model.Maxpool(x1)
            x2 = model.Conv2(x1_p)
            x2_p = model.Maxpool(x2)
            x3 = model.Conv3(x2_p)
            x3_p = model.Maxpool(x3)
            x4 = model.Conv4(x3_p)
            x4_p = model.Maxpool(x4)
            x5 = module(x4_p)
            if name in self.target_layers:
                x5.register_forward_hook(forward_hook)
                x5.register_backward_hook(backward_hook)
                x5.register_hook(self.save_gradient)
                outputs += [x5]
            d5 = model.Up5(x5)
            x4_att = model.Att5(d5, x4)
            d5 = torch.cat((x4_att, d5), dim=1)
            d5 = model.Up_conv5(d5)

            d4 = model.Up4(d5)
            x3_att = model.Att4(d4, x3)
            d4 = torch.cat((x3_att, d4), dim=1)
            d4 = model.Up_conv4(d4)

            d3 = model.Up3(d4)
            x2_att = model.Att3(d3, x2)
            d3 = torch.cat((x2_att, d3), dim=1)
            d3 = model.Up_conv3(d3)

            d2 = model.Up2(d3)
            
            return outputs, d2

class Feature:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        
        self.model = model.cuda()

        self.extractor = FeatureExtractor(self.model, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)
    
    def __call__(self,block,input_img):
        input_img = input_img.cuda()
        features, output = self.extractor(input_img)
        #print(self.extractor.get_gradients())
        #a = self.extractor.get_gradients()
        #print(a)

def forward(tensor_img_NAC,tensor_img_AC,model,device):
    tensor_img_NAC = img_transform(tensor_img_NAC,device)
    tensor_img_AC = img_transform(tensor_img_AC,device)
    output = model(tensor_img_NAC)
    return output

def img_transform(img,device):
    img = img.float()
    img = img.to(device=device)
    return img

def backward_hook(module,grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# get feature maps
def forward_hook(module,input,output):
    fmap_block.append(output)



def backward(loss_function,model,output,tensor_img_AC,device):
    tensor_img_AC = img_transform(tensor_img_AC,device)
    loss = loss_function(output,tensor_img_AC) 
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    optimizer.zero_grad()
    loss.backward()   

def gen_cam(feature_map,grads):
    cam = np.zeros(feature_map.shape[1:],dtype=np.float32) # 384x384
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = cv2.resize(cam, (384, 384))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def show_cam_on_image(img, cam, out_dir,time_stamp,model_name,slice_id):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) 
    path_out = os.path.join(out_dir, str(time_stamp)+"_"+"attention_maps"+"_"+str(model_name)+"_"+str(slice_id)+".tif")

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET) # do heatmap must with uint8
    cam_heatmap = np.float32(cam_heatmap)/255

    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cam_img_heatmap = 0.3*cam_heatmap + 0.7*np.float32(img_rgb)/255
    cam_img_heatmap = cam_img_heatmap / np.max(cam_img_heatmap)

    fig = plt.figure(figsize=(21,5)) 
    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray') 
    plt.title('img_NAC')

    plt.subplot(1,3,2)
    plt.imshow(cam_heatmap,'jet') 
    plt.colorbar()
    plt.title('attention maps of '+str(model_name))

    plt.subplot(1,3,3)
    plt.imshow(cam_img_heatmap,'jet')
    plt.colorbar()
    plt.title('attention maps and img_NAC')
    plt.show()
    fig.savefig(path_out, bbox_inches='tight')

def store_cam(tensor_img_NAC,output_dir,time_stamp,model_name,slice_id):
    grads_val = grad_block[0].cpu().data.numpy().squeeze(0)
    fmap = fmap_block[0].cpu().data.numpy().squeeze(0)
    cam = gen_cam(fmap, grads_val)
    cam = np.float32(resize_sitk_2D(cam, (344, 344)))
    tensor_img_NAC = tensor_img_NAC.cpu().data.numpy().squeeze()
    tensor_img_NAC = np.float32(resize_sitk_2D(tensor_img_NAC, (344, 344)))
    show_cam_on_image(tensor_img_NAC, cam, output_dir,time_stamp,model_name,slice_id)
    print('Attention Map of',str(model_name),'with slice_id',str(slice_id),'is saved!')

    
if __name__ == '__main__':

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = '/workspace/smbohann/Unettry/attention_maps_attention_unet'
    patient_list = ['Patient15'] 
    data_path ='/workspace/smbohann/Unettry/mydataset'
    slice_list = [50, 140, 300]
    slice_id = 140
    path_NAC = os.path.join(data_path,patient_list[0],patient_list[0]+"_NAC.gipl")
    tensor_img_NAC = load_data(path_NAC,slice_id)
    #print(tensor_img_NAC.shape)
    path_AC = os.path.join(data_path,patient_list[0],patient_list[0]+"_AC.gipl")
    tensor_img_AC = load_data(path_AC,slice_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_function = nn.MSELoss()
    grad_block = []
    fmap_block = []

    model = att_unet()
    model_path = '/workspace/smbohann/Unettry/checkpoint/best_model_2021-02-02_18:43:07.pth'
    
    load_model(model,model_path,device)
    model.Att2.register_forward_hook(forward_hook)
    model.Att2.register_backward_hook(backward_hook)
    forw = Feature(model=model,target_layer_names=["3"])
    input_img = tensor_img_NAC
    forw(block=fmap_block,input_img=input_img)
    output = forward(tensor_img_NAC,tensor_img_AC,model,device)
    model.Up_conv2.register_forward_hook(forward_hook)
    model.Up_conv2.register_backward_hook(backward_hook)
    print(grad_block)
    #backward(loss_function,model,output,tensor_img_AC,device)
    forw(block=grad_block,input_img=input_img)
    #store_cam(tensor_img_NAC,output_dir,time_stamp,'unet',slice_id)

    
    for name, module in model.Conv5.conv._modules.items():
        x = tensor_img_NAC
        x = x.float()
        x = x.to(device=device)
            
        x1 = model.Conv1(x)
        x1_p = model.Maxpool(x1)
        x2 = model.Conv2(x1_p)
        x2_p = model.Maxpool(x2)
        x3 = model.Conv3(x2_p)
        x3_p = model.Maxpool(x3)
        x4 = model.Conv4(x3_p)
        x4_p = model.Maxpool(x4)
        print(x4_p.shape)
        x4_p = x4_p.expand(1, 1024, 24, 24)
        print(x4_p.shape)
        if name in ['3']:
            x5_3 = module(x5_2)
            x5_3.register_forward_hook(forward_hook)
            x5_3.register_backward_hook(backward_hook)
        x5_4 = module(x5_3)
        x5 = module(x5_4)
        x5 = model.Conv5(x4_p)
        d5 = model.Up5(x5)
        x4_att = model.Att5(d5, x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = model.Up_conv5(d5)

        d4 = model.Up4(d5)
        x3_att = model.Att4(d4, x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = model.Up_conv4(d4)

        d3 = model.Up3(d4)
        x2_att = model.Att3(d3, x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = model.Up_conv3(d3)

        d2 = model.Up2(d3)
        print(grad_block)
        

        
       


     
   