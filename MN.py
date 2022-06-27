# -*- coding: utf-8 -*-
"""
@author: Shuang Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

## Residual Block
class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

## Depthwise separable conv
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin,  3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, 1)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

## Reweight Block
class ReweightBlock(nn.Module):
    def __init__(self, n_channels, reduction):
        super(ReweightBlock, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        r_channels = max(8, n_channels // reduction)
        self.conv_down = nn.Sequential(nn.Conv2d(n_channels, r_channels, 1), nn.ReLU(True))
        self.conv_h = nn.Conv2d(r_channels, n_channels, 1)
        self.conv_w = nn.Conv2d(r_channels, n_channels, 1)
    
    def forward(self, x):
        _,_,h,w = x.shape
        
        x_h = self.pool_h(x) # [N,C,H,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [N,C,W,1]
        y = torch.cat([x_h, x_w], dim=2) # [N,C,H+W,1]
        y = self.conv_down(y) # [N,C/r,H+W,1]
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        weight = a_h*a_w
        return weight

## Branch Selector
class BranchSelector(nn.Module):
    def __init__(self, n_branch, n_channels, reduction):
        super(BranchSelector, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Sequential(nn.Conv2d(n_channels, n_channels // reduction, 1, bias=False), nn.ReLU(inplace=True))
        self.conv_up = nn.ModuleList([nn.Sequential(nn.Conv2d(n_channels // reduction, n_channels, 1, bias=False), nn.Sigmoid()) for _ in range(n_branch)])
            
    def forward(self, x):
        # x - [N,C,H,W,B]. The last dimension is the number of branches
        y = torch.sum(x, dim=-1)
        y = self.avg_pool(y)
        y = self.conv_down(y)
        
        weight = []
        for i in range(len(self.conv_up)):
            weight.append(self.conv_up[i](y))
        weight = torch.stack(weight, dim=-1)
        weight = F.softmax(weight, dim=-1)
        return torch.sum(x*weight, dim=-1)

## Gaussian Branch
class GaussianBranch(nn.Module):
    def __init__(self, n_feat, init_lambd, init_rho):
        super(GaussianBranch, self).__init__()
        self.lambd = nn.Parameter(torch.full(size = (1, n_feat, 1, 1), fill_value=init_lambd))
        self.rho   = nn.Parameter(torch.full(size = (1, n_feat, 1, 1), fill_value=init_rho))
        
    def forward(self, T, G, Z, U):
        output = T + self.lambd * G + self.rho * (Z - U)
        output = output / (1 + self.lambd + self.rho)
        return output

## NonGaussian Branch
class NonGaussianBranch(nn.Module):
    def __init__(self, n_feat, reduction, init_rho):
        super(NonGaussianBranch, self).__init__()
        self.dsc_X = nn.Sequential(DepthwiseSeparableConv(n_feat, n_feat), nn.ReLU(True)) # depthwise separable conv
        self.dsc_G = nn.Sequential(DepthwiseSeparableConv(n_feat, n_feat), nn.ReLU(True)) # depthwise separable conv
        self.dsc_B = nn.Sequential(DepthwiseSeparableConv(n_feat, n_feat), nn.ReLU(True)) # depthwise separable conv
        self.reweight = ReweightBlock(n_feat, reduction)
        
        self.rho   = nn.Parameter(torch.full(size = (1, n_feat, 1, 1), fill_value=init_rho))
        
    def forward(self, T, G, X, Z, U):
        _F = self.dsc_X(X) + self.dsc_G(G)
        W = self.reweight(_F)
        B = self.dsc_B(_F)
        
        output = T + W*(G-B) + self.rho*(Z-U)
        output = output / (1+W+self.rho)
        return output 


## XNet
class XNet(nn.Module):
    def __init__(self, n_branch, n_feat, reduction, init_lambd, init_rho):
        super(XNet, self).__init__()
        self.branch_Gauss = GaussianBranch(n_feat, init_lambd, init_rho)
        self.branch_NonGauss = nn.ModuleList([NonGaussianBranch(n_feat, reduction, init_rho) for _ in range(n_branch)])
        self.branch_selector = BranchSelector(n_branch+1, n_feat, reduction)
        
    def forward(self, T,G,X,Z,U):
        output = []
        output.append(self.branch_Gauss(T,G,Z,U))
        for i in range(len(self.branch_NonGauss)):
            output.append(self.branch_NonGauss[i](T,G,X,Z,U))
            
        output = torch.stack(output, dim=-1) # [N,C,H,W,n_branch+1]
        output = self.branch_selector(output)
        return output    

## MN: Mixture Network
class MN(nn.Module):
    def __init__(self, 
                 target_channels,
                 guidance_channels,
                 n_layer    = 7,
                 n_feat     = 64,
                 n_branch   = 2,
                 reduction  = 8,
                 init_lambd = 1e-1,
                 init_rho   = 1e-1):
        super(MN, self).__init__()
        
        self.n_layer = n_layer
        
        # Encoder
        self.conv0_T = nn.Sequential(nn.Conv2d(target_channels, n_feat, 3, padding=1), nn.ReLU(True))
        self.conv0_G = nn.Sequential(nn.Conv2d(guidance_channels, n_feat, 3, padding=1), nn.ReLU(True))
        
        # Decoder
        self.recon = nn.Conv2d(n_feat, target_channels, 3, padding=1)
        
        # Body - XNet
        self.x_net = nn.ModuleList([XNet(n_branch, n_feat, reduction, init_lambd, init_rho) for _ in range(n_layer)])
        
        # Body - ProxNet        
        self.res_blocks = nn.ModuleList([ResBlock(n_feat) for _ in range(n_layer)])
        
    def forward(self, T, G):
        T = self.conv0_T(T)
        G = self.conv0_G(G)
        
        Z = torch.zeros_like(T)
        U = torch.zeros_like(T)
        X = 'none'
        
        for i in range(self.n_layer):
            # X update
            if X == 'none':
                X = self.x_net[i](T,G,T,Z,U)
            else:
                X = self.x_net[i](T,G,X,Z,U)
            # Z update
            Z = self.res_blocks[i](X + U)
            # U update
            U = U + X - Z
        
        return self.recon(Z)