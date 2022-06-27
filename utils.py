# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:43:42 2020

@author: win10
"""

# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------

import os
import json

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
        
def save_param(input_dict, path):
    f = open(path, 'w')
    f.write(json.dumps(input_dict))
    f.close()
    print("Hyper-Parameters have been saved!")
    

# ----------------------------------------------------------------------------
# Dataset & Image Processing
# ----------------------------------------------------------------------------

import h5py
import torch
import numpy as np
import torch.utils.data as Data
from skimage.io import imread
    
def im2double(img):
    if img.dtype=='uint8':
        img = img.astype(np.float32)/255.
    elif img.dtype=='uint16':
        img = img.astype(np.float32)/65535.
    else:
        img = img.astype(np.float32)
    return img

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def imresize(img, size=None, scale_factor=None):
    # img (np.array) - [C,H,W]
    imgT = torch.from_numpy(img).unsqueeze(0) #[1,C,H,W]
    if size is None and scale_factor is not None:
        imgT = torch.nn.functional.interpolate(imgT, scale_factor=scale_factor)
    elif size is not None and scale_factor is None:
        imgT = torch.nn.functional.interpolate(imgT, size=size)
    else:
        print('Neither size nor scale_factor is given.')
    imgT = imgT.squeeze(0).numpy()
    return imgT
    
def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist

def prepare_data(data_path, 
                 patch_size, 
                 aug_times=4,
                 stride=128, 
                 file_name='train.h5'
                 ):

    print('process training data')
    T_files = get_img_file(os.path.join(data_path, 'target'))
    G_files = get_img_file(os.path.join(data_path, 'guidance'))
    
    h5f        = h5py.File(file_name, 'w')
    h5gt       = h5f.create_group('GT')
    h5guidance = h5f.create_group('Guidance')
    
    train_num = 0
    for i in range(len(T_files)):
        I_gt       = imread(T_files[i])
        I_guidance = imread(G_files[i])
        if len(I_gt.shape)==2:
            I_gt = I_gt[:,:,None]
        if len(I_guidance.shape)==2:
            I_guidance = I_guidance[:,:,None]
        I_gt       = np.transpose(I_gt.astype('float32'), [2,0,1])/255. # [Height, Width, Channels]
        I_guidance = np.transpose(I_guidance.astype('float32'), [2,0,1])/255. # [Height, Width, Channels]
     
        gt_patches       = Im2Patch(I_gt,       win=patch_size, stride=stride) #[C,H,W,N]
        guidance_patches = Im2Patch(I_guidance, win=patch_size, stride=stride)
  
        print("file: %s # samples: %d" % (T_files[i], gt_patches.shape[3]*aug_times))
        for n in range(gt_patches.shape[3]):
            # GT and Guidance
            gt_data       = gt_patches[:,:,:,n].copy()
            guidance_data = guidance_patches[:,:,:,n].copy()
            
            # Write data into H5 file
            h5gt.create_dataset(str(train_num), 
                                data=gt_data,       dtype=gt_data.dtype,       shape=gt_data.shape)
            h5guidance.create_dataset(str(train_num), 
                                data=guidance_data, dtype=guidance_data.dtype, shape=guidance_data.shape)
            
            train_num += 1
            for m in range(aug_times-1):
                gt_data_aug       = np.rot90(gt_data, m+1, axes=(1,2))
                guidance_data_aug = np.rot90(guidance_data, m+1, axes=(1,2))
                
                h5gt.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                    data=gt_data_aug,       dtype=gt_data_aug.dtype,       shape=gt_data_aug.shape)
                h5guidance.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                    data=guidance_data_aug, dtype=guidance_data_aug.dtype, shape=guidance_data_aug.shape)
                train_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    
class CMD_Dataset(Data.Dataset):
    def __init__(self, root, with_gt=True):
        self.with_gt = with_gt
        self.root = root
        self.guidance_files = get_img_file(os.path.join(root, 'guidance'))
        self.target_files   = get_img_file(os.path.join(root, 'target'))
        if with_gt:
            self.gt_files   = get_img_file(os.path.join(root, 'gt'))

    def __len__(self):
        return len(self.guidance_files)
    
    def __getitem__(self, index):
        I_guidance = imread(self.guidance_files[index])
        I_target   = imread(self.target_files[index])
        if len(I_target.shape)==2:
            I_target = I_target[:,:,None]
        if len(I_guidance.shape)==2:
            I_guidance = I_guidance[:,:,None]
            
        I_guidance = np.transpose(I_guidance.astype('float32'), [2,0,1])/255.
        I_target   = np.transpose(I_target.astype('float32'), [2,0,1])/255.
        
        if self.with_gt:
            I_gt = imread(self.gt_files[index])
            I_gt = np.transpose(I_gt.astype('float32'), [2,0,1])/255.
            return I_target, I_guidance, I_gt
        else:
            return I_target, I_guidance

class CMD_H5Dataset(Data.Dataset):
    def __init__(self, h5file_path, sigma_min=0., sigma_max=75.):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['Target'].keys())
        h5f.close()
        self.sigma_min = sigma_min/255.
        self.sigma_max = sigma_max/255.

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        gt       = np.array(h5f['GT'][key])
        guidance = np.array(h5f['Guidance'][key])
        h5f.close()
        
        temp_sigma  = float(np.random.rand(1)*(self.sigma_max-self.sigma_min)+self.sigma_min)
        target = gt + np.random.randn(*gt.shape)*temp_sigma
        
        return torch.Tensor(target),torch.Tensor(guidance),torch.Tensor(gt)