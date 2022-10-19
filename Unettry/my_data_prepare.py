import SimpleITK as sitk
import os
import time

if __name__ == '__main__':
    # "/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl/Patient01/Patient01_AC.gipl"
    patient_list = ['Patient01','Patient02','Patient05', 'Patient06','Patient12','Patient15']
    original_path = '/workspace/MedIP-PP/WS20_PMB/NAC_AC_gipl' # original dataset
    new_path = '/workspace/smbohann/Unettry/my2' # renew dataset (only obtains the head of each patient), to accelerate training 

    for p in patient_list:

        path_NAC = os.path.join(original_path,p,p+"_NAC.gipl")
        path_AC = os.path.join(original_path,p,p+"_AC.gipl")

        path_sv_NAC = os.path.join(new_path,p,p+"_NAC.gipl") #sv= small volume
        path_sv_AC = os.path.join(new_path,p,p+"_AC.gipl")
        correct = sitk.N4BiasFieldCorrectionImageFilter()
        # read gipl and get part
        sitk_vol1 = sitk.ReadImage(path_AC)
         # tensor
        sv_sitk_vol = sitk_vol1[:,:,:]
        mask_image2 = sitk.OtsuThreshold(sv_sitk_vol ,0,1,200)
        outputimage2 = correct.Execute(sv_sitk_vol,mask_image2) # get 5-45. slices/images from gipl 0-417(0-515)， 344,344,418(sequence of dimensions changed)

        # change gipl to array
        np_vol = sitk.GetArrayFromImage(sitk_vol1)
        np_svol = sitk.GetArrayFromImage(outputimage2)
        print('shape of np_vol:', np_vol.shape)
        print('shape of sitk_svol:', np_svol.shape)
    
        # save the obtain small volume as new gipl
        sitk.WriteImage(sv_sitk_vol, path_sv_AC)

        # do same operation on NAC-PET
        sitk_vol = sitk.ReadImage(path_NAC)
        sv_sitk_vol = sitk_vol[:,:,:]
        mask_image1 = sitk.OtsuThreshold(sv_sitk_vol,0,1,200)
        outputimage1 = correct.Execute(sv_sitk_vol,mask_image1)
        np_vol = sitk.GetArrayFromImage(sitk_vol )
        np_svol = sitk.GetArrayFromImage(outputimage1)
        print(np_vol.shape,np_svol.shape)
        sitk.WriteImage(sv_sitk_vol, path_sv_NAC)

        # check the gotten small volume 
        print('path_sv_NAC:',path_sv_NAC)
        print('pathe_sc_AC:',path_sv_AC)
