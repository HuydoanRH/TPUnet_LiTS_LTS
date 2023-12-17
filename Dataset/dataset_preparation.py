import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
from os.path import join
import config
import SimpleITK as sitk
import random
import skimage.transform  as sktf
import glob
import matplotlib.pyplot as plt
class LITS_slicing_train_val:
    def __init__(self, raw_dataset_root_path,slice_dataset_root_path, args):
        self.raw_dataset_root_path = raw_dataset_root_path
        self.slice_dataset_root_path = slice_dataset_root_path
        self.valid_rate = args.valid_rate
        self.fillters = args.fillters
        self.rs_shape = (128,128,1)
        self.tumor_label = True
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice
        self.size = args.size

    def slice_data(self):
        if not os.path.exists(self.slice_dataset_root_path):
            os.makedirs(join(self.slice_dataset_root_path,'ct'))
            os.makedirs(join(self.slice_dataset_root_path, 'label'))

        file_list = sorted(os.listdir(join(self.raw_dataset_root_path,'ct')))
        # file_list = ['volume-0.nii']
        Numbers = len(file_list)
        print('Total numbers of samples is :',Numbers)
        for ct_file,i in zip(file_list,range(Numbers)):
            print("********* {} | {}/{} *********".format(ct_file, i+1,Numbers))
            ct_path = os.path.join(self.raw_dataset_root_path, 'ct', ct_file)
            seg_path = os.path.join(self.raw_dataset_root_path, 'label', ct_file.replace('volume', 'segmentation'))
            self.process(ct_file, ct_path, seg_path, self.rs_shape)

    def process(self, ct_file, ct_path, seg_path, rs_shape):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        print("Ori shape:",ct_array.shape, seg_array.shape)

        if self.tumor_label:
            seg_array[seg_array <= 1] = 0
            seg_array[seg_array > 1] = 1

        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower

        z = np.any(seg_array, axis=(1, 2))
        print(z.shape)
        try:
            start_slice, end_slice = np.where(z)[0][[0, -1]]
        except:
            print ("Don't have tumor at this volume")
            return

        if start_slice - self.expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1
        else:
            end_slice += self.expand_slice

        print("Cut out range:",str(start_slice) + '--' + str(end_slice))
        if end_slice - start_slice + 1 < self.size:
            print('Too little slice，give up the sample:', ct_path)
            return None,None

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        # ct_slice = ct_array[85,:,:]
        # np.save('test.npy', ct_slice)
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        print("Preprocessed shape:",ct_array.shape,seg_array.shape)
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())

        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        sum_tumor_pixel = 0
        num_slice = 0
        ct_array = sitk.GetArrayFromImage(new_ct)
        deep_ct_array = ct_array.shape[0]

        seg_array = sitk.GetArrayFromImage(new_seg)
        deep_seg_array = seg_array.shape[0]

        print("Ori shape volume after filter:",ct_array.shape, seg_array.shape)
        print("Ori deep volume after filter:",deep_ct_array, deep_seg_array)

        for i in range(0, deep_ct_array):
            sum_tumor_pixel = np.sum(seg_array[i,:,:] > 0)
            print("==== sum_tumor_pixel of slice {} of {} = {} ====".format(i, ct_file, sum_tumor_pixel))
            if(sum_tumor_pixel < self.fillters):
                continue
            else:
                print("==== slicing {} | {}/{} ====".format(ct_file, num_slice, deep_seg_array))
                ct_rs = sktf.resize(ct_array[i,:,:], rs_shape, mode = 'constant', preserve_range = True)
                ct_rs = (ct_rs - ct_rs.min())/((ct_rs.max()-ct_rs.min()))
                ct_rs = ct_rs.astype(np.float32)

                label_rs = sktf.resize(seg_array[i,:,:], rs_shape, mode = 'constant', preserve_range = True)
                label_rs = label_rs > 0.5
                label_rs = label_rs.astype(np.float32)
                path_slice_ct = os.path.join(self.slice_dataset_root_path, 'ct', ct_file.replace('.nii', f"_slice_{num_slice}.npy"))
                path_slice_label = os.path.join(self.slice_dataset_root_path, 'label', ct_file.replace('volume', 'segmentation').replace('.nii', f"_slice_{num_slice}.npy"))
                np.save(path_slice_ct, ct_rs)
                print("slice ct shape:",ct_rs.shape)
                np.save(path_slice_label, label_rs)
                print("slice label shape:", label_rs.shape)
                num_slice += 1

    def count_num_slice(self):
        slice_name_list_ct = os.listdir(join(self.slice_dataset_root_path, "ct"))
        slice_num = len(slice_name_list_ct)
        print('the sliced total numbers of samples ct is :', slice_num)
        slice_name_list_label = os.listdir(join(self.slice_dataset_root_path, "label"))
        slice_num = len(slice_name_list_label)
        print('the sliced total numbers of samples label is :', slice_num)

if __name__ == '__main__':
    raw_dataset_path = './raw_dataset/train/'
    fixed_dataset_path = './fixed_dataset/'
    args = config.args 
    tool = LITS_slicing_train_val(raw_dataset_path,fixed_dataset_path, args)
    tool.slice_data()
    tool.count_num_slice()