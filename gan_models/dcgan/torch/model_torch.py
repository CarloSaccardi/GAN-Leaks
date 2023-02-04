from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np




def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(mode, size):
    if mode == 'normal01': return np.random.normal(0, 1, size=size)
    if mode == 'uniform_signed': return np.random.uniform(-1, 1, size=size)
    if mode == 'uniform_unsigned': return np.random.uniform(0, 1, size=size)
    
    
    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input is N x (nc) x 64 x 64
            nn.Conv2d(input_size, hidden_size, kernel_size = 4, stride = 2, padding = 1, bias=False), # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(hidden_size, hidden_size * 2, 4, 2, 1),# 16x16
            self._block(hidden_size * 2, hidden_size * 4, 4, 2, 1),# 8x8
            self._block(hidden_size * 4, hidden_size * 8, 4, 2, 1),# 4x4
            nn.Conv2d(hidden_size * 8, output_size, kernel_size = 4, stride = 2, padding = 0, bias=False), # 1x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
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
            
        