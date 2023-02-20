from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
    
    
class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input is N x (nc) x 64 x 64
            nn.Conv2d(img_channels, features_d, kernel_size = 4, stride = 2, padding = 1), # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),# 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),# 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),# 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size = 4, stride = 2, padding = 0), # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),#similar to layernorm 
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.disc(x)
    
    
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            #input is N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 1, 0), # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1), # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Tanh(),# [-1, 1]
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.gen(x)
    
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            


if __name__ == '__main__':
    N, in_channels, H, W = 8, 3, 64, 64
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8, 1)
    initialize_weights(disc)
    print(disc(x).shape)
    z_dim = 100
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    print(gen(z).shape)          
            
        