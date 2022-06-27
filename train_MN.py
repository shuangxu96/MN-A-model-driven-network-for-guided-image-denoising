# -*- coding: utf-8 -*-
"""
@author: Shuang Xu
"""

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

import os
import time
import datetime

import torch
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader 
from glob import glob

import sys
# sys.path.append("..")
from MN import MN as Net
from metrics import PSNRLoss, SSIM
from utils import CMD_H5Dataset, CMD_Dataset, prepare_data

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

sigma_max, sigma_min = 75., 0.
model_str = 'MN'
data_str = 'NIR' # only support 'NIR' (RNS) or 'Flash' (FAIP)
light_weight = False # if use light weight version, there are 3 layers and 32 filters

if light_weight:
    n_layer, n_feat = 3, 32
else:
    n_layer, n_feat = 7, 64

# . Get the parameters of the dataset
if data_str=='NIR':
    target_channels = 3 # the number of channels for RGB images
    guidance_channels = 1 # the number of channels for NIR images
elif data_str=='Flash':
    target_channels = 3 # the number of channels for non-flash images
    guidance_channels = 3 # the number of channels for flash images

# . Set the hyper-parameters for training
num_epochs = 100
lr = 5e-4
weight_decay = 0
batch_size = 8


# . Get your model 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net(target_channels,
          guidance_channels,
          n_layer = n_layer,
          n_feat  = n_feat
           ).to(device)
print(net)

# . Get your optimizer, scheduler and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.L1Loss().to(device)
psnr = PSNRLoss(max_val=1.)
ssim = SSIM(window_size=11, reduction='mean', max_val=1.)

# . Create your data loaders
prepare_data_flag = False # If this is the first time to run this code, please
                          # set it to True, and it will generate an .h5 file
                          # containing training data. Once you have run this 
                          # code, plz set it to False, bacause the .h5 file 
                          # has been generated in the first run.
train_path      = './dataset/%s/%s_train.h5'%(data_str,data_str)
validation_path = './dataset/%s/val'%(data_str)
if prepare_data_flag is True:
    prepare_data(data_path = './dataset/%s/train'%(data_str), 
                 patch_size=128, aug_times=1, stride=128, file_name = train_path)

trainloader      = DataLoader(CMD_H5Dataset(train_path), 
                              batch_size=batch_size, 
                              shuffle=True) #[N,C,K,H,W]
validationloader = DataLoader(CMD_Dataset(validation_path,False),      
                              batch_size=1)

test_path  = glob('./dataset/%s/test*'%(data_str))
testloader = {}
for path in test_path:
    testloader[path.split('test_')[-1]] = DataLoader(CMD_Dataset(path), batch_size=1)

# . Creat logger
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join(
    'logs/%s'%(model_str),
    timestamp+'_%s_%dfilters_%dlayers'%(
               data_str,n_feat,n_layer)
    )
writer = SummaryWriter(save_path)

import shutil
shutil.copy('train_MN.py', os.path.join(save_path,'train_MN.py'))
shutil.copy('MN.py',       os.path.join(save_path,'MN.py'))

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
best_psnr_val,psnr_val, ssim_val = 0., 0., 0.
torch.backends.cudnn.benchmark = True
prev_time = time.time()


for epoch in range(num_epochs):
    ''' train '''
    for i, (target, guidance, gt) in enumerate(trainloader):
        # 0. preprocess data
        target, guidance, gt = target.cuda(), guidance.cuda(), gt.cuda()
        
        # 1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        output = net(target, guidance)
        loss = loss_fn(output, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1000, norm_type=2)
        optimizer.step() 
        
        # 2. print
        # Determine approximate time left
        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [PSNR/Best: %.4f/%.4f] ETA: %s"
            % (
                epoch,
                num_epochs,
                i,
                len(trainloader),
                loss.item(),
                psnr_val,
                best_psnr_val,
                time_left,
            )
        )

        # 3. Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step+=1
        
    ''' validation ''' 
    current_psnr_val = psnr_val
    psnr_val = 0.
    ssim_val = 0.
    with torch.no_grad():
        net.eval()
        for i, (target, guidance) in enumerate(validationloader):
            target, guidance = target.cuda(), guidance.cuda()
            temp_sigma  = float(np.random.rand(1)*(sigma_max-sigma_min)+sigma_min)
            output = net(target+torch.randn_like(target)*temp_sigma/255., guidance)
            psnr_val += psnr(output, target)
            ssim_val += ssim(output, target)
        psnr_val = float(psnr_val/validationloader.__len__())
        ssim_val = float(ssim_val/validationloader.__len__())
    writer.add_scalar('PSNR/val', psnr_val, epoch)
    writer.add_scalar('SSIM/val', ssim_val, epoch)

    ''' save model ''' 
    # Save the best weight
    if best_psnr_val<psnr_val:
        best_psnr_val = psnr_val
        torch.save({'net':net.state_dict(), 
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch},
                   os.path.join(save_path, 'best_net.pth'))
    # Save the current weight
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))
    
    ''' backtracking '''
    if epoch>0:
        if torch.isnan(loss):
            print(10*'='+'Backtracking!'+10*'=')
            net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['net'])
            optimizer.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['optimizer'])


print('End')                                                                      