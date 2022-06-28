# -*- coding: utf-8 -*-
"""
@author: Shuang Xu
"""

import os
import torch

from skimage.io import imsave, imread
from utils import get_img_file


def load_img(path):
    I = imread(path)
    I = torch.from_numpy(I).cuda().float()/255.
    if len(I.shape)==2:
        I = I[:,:,None]
    I = I.permute(2,0,1)[None,:,:,:]
    return I

model_str = 'MN' 
data_str = 'Flash'
light_weight = True

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
save_path = 'output_eval/%s_%s'%(model_str, data_str)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load weight
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net(target_channels, guidance_channels, n_layer, n_feat).to(device)
weight_path = 'weight/%s_%s.pth'%(model_str, data_str)
net.load_state_dict(torch.load(weight_path)['net'])
print(net)

# set test images
g_files = get_img_file(r'eval/%s/guidance/'%data_str)
t_files = get_img_file(r'eval/%s/target/'%data_str)

# write denoised images
with torch.no_grad():
    net.eval()
    print('Start to write denoised images.')
    for i in range(len(g_files)):
        guidance = load_img(g_files[i]).to(device)
        target = load_img(t_files[i]).to(device)
        output = net(target, guidance).clamp(0.,1.)
        output = output.squeeze(0).detach().permute(1,2,0).cpu().numpy()*255.
        output = output.astype('uint8')
        temp_name = t_files[i].split('/')[-1]
        imsave(os.path.join(save_path,temp_name),
                output
                )
