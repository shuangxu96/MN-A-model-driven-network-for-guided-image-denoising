# -*- coding: utf-8 -*-
"""
@author: Shuang Xu
"""

import os
import xlwt

import torch
import numpy as np

from metrics import PSNRLoss, SSIM
from torch.utils.data import DataLoader 
from skimage.io import imsave, imread
from glob import glob
from utils import CMD_Dataset


def load_img(path):
    I = imread(path)
    I = torch.from_numpy(I).cuda().float()/255.
    I = I.permute(2,0,1)[None,:,:,:]
    return I

model_str = 'NGN' 
data_str = 'NIR'
light_weight = False

# import network
if model_str=='MN':
    from MN import MN as Net

# set network parameters (depending on light weight or not)
if light_weight:
    model_str = model_str+'L'
    n_layer, n_feat = 3, 32
else:
    n_layer, n_feat = 7, 64

# set dataset channels
if data_str=='NIR':
    target_channels, guidance_channels = 3, 1
elif data_str=='Flash':
    target_channels, guidance_channels = 3, 3

# set the output folder
save_path = 'output/%s_%s'%(model_str, data_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load weight
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net(target_channels, guidance_channels, n_layer, n_feat).to(device)
weight_path = 'weight/%s_%s.pth'%(model_str, data_str)
net.load_state_dict(torch.load(weight_path)['net'])
print(net)

# define metrics
psnr = PSNRLoss(max_val=1.)
ssim = SSIM(window_size=11, reduction='mean', max_val=1.)

# set test images
test_path  = glob('dataset/%s/test*'%(data_str))
testloader = {}
for path in test_path:
    testloader[path.split('test_')[-1]] = DataLoader(CMD_Dataset(path), batch_size=1)

# write denoised images
with torch.no_grad():
    net.eval()
    print('Start to write denoised images.')
    for temp_key, temp_loader in testloader.items():
        for i, (target, guidance, gt) in enumerate(temp_loader):
            target, guidance = target.cuda(), guidance.cuda()
            output = net(target, guidance).clamp(0.,1.)
            output = output.squeeze(0).detach().permute(1,2,0).cpu().numpy()*255.
            output = output.astype('uint8')
            temp_name = temp_loader.dataset.target_files[i].split('\\')[-1].split('.')[0] + '.png'
            imsave(os.path.join(save_path,temp_key+temp_name),
                    output
                    )

# load denoised images and calculate metrics
metrics = {}
print('Start to calculate metrics.')
for temp_key, temp_loader in testloader.items():
    temp_metrics = torch.zeros(2,temp_loader.__len__())
    files = temp_loader.dataset.target_files
    for i, (target, guidance, gt) in enumerate(temp_loader):
        gt = gt.cuda()
        
        prefix = files[i].split('\\')[-3].replace('test_','')
        subfix = files[i].split('\\')[-1].replace('tiff','png')
        output = load_img(os.path.join(save_path,prefix+subfix))
        
        temp_metrics[0,i] = psnr(output, gt)
        temp_metrics[1,i] = ssim(output, gt)
        temp_name = temp_loader.dataset.target_files[i].split('\\')[-1].split('.')[0] + '.png'
    metrics[temp_key] = temp_metrics

# Write the metrics
print('Start to write metrics.')
f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
metric_name = ['PSNR','SSIM']

# Main table
row = 0
for temp_key, temp_metrics in metrics.items():
    # 1st row
    sheet1.write(row, 0, temp_key) # write the testset name
    img_name = [i.split('\\')[-1] for i in testloader[temp_key].dataset.target_files]
    for j in range(len(img_name)):
        sheet1.write(row, j+1, img_name[j])  
    
    # metric rows
    for i in range(len(metric_name)):
        for j in range(len(img_name)):
            sheet1.write(row+i+1, j+1, float(temp_metrics[i,j]))
            
    # metric row header
    for i in range(len(metric_name)):
        sheet1.write(row+i+1,0,metric_name[i])

    row += len(metric_name)+1 

# Mean table
row += 2
sheet1.write(row,0,'Mean')
col = 1

a25, a50, a75 = np.zeros(len(metric_name)), np.zeros(len(metric_name)), np.zeros(len(metric_name))
for temp_key, temp_metrics in metrics.items():
    sheet1.write(row, col, temp_key)
    mean_metrics = temp_metrics.mean(dim = 1)
    if col%3==1:
        for i in range(len(metric_name)):
            a25[i] += float(mean_metrics[i])
    elif col%3==2:
        for i in range(len(metric_name)):
            a50[i] += float(mean_metrics[i])
    elif col%3==0:
        for i in range(len(metric_name)):
            a75[i] += float(mean_metrics[i])
        
    for i in range(len(metric_name)):
        sheet1.write(row+i+1, col, float(mean_metrics[i]))
    col += 1

a25 /= len(metrics)/3
a50 /= len(metrics)/3
a75 /= len(metrics)/3
sheet1.write(row, col,   'A25')
sheet1.write(row, col+1, 'A50')
sheet1.write(row, col+2, 'A75')
for i in range(len(metric_name)):
    sheet1.write(row+i+1, col,   float(a25[i]))
for i in range(len(metric_name)):
    sheet1.write(row+i+1, col+1, float(a50[i]))
for i in range(len(metric_name)):
    sheet1.write(row+i+1, col+2, float(a75[i]))

    
# Save
f.save(os.path.join(save_path,'1test_result.xls'))