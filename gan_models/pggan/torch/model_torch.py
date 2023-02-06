import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):   #weight scaled conv layer, EQULIZED LEARNING RATE
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2): #gain is for the initialization constant
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5 
        self.bias = self.conv.bias 
        self.conv.bias = None #remove the bias from the conv layer cause it won't be scaled
        
        #initialize the weights
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        #scale the input and add the bias and reshape it
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1) 
        

class PixelNorm(nn.Module):#pixel normalization layer across the channels
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

class ConvBlock(nn.Module): # 3x3 conv block with pixel norm and leaky relu
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixel_norm
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.pn(x) if self.use_pn else x
        
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.pn(x) if self.use_pn else x
        
        return x
        
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),# 1x1 conv transpose to 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        
        self.initial_rgb = nn.Conv2d(in_channels, img_channels, kernel_size=1, stride=1)
        
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList(self.initial_rgb)
        
        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(nn.Conv2d(conv_out_channels, img_channels, kernel_size=1, stride=1))
        
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)
    
    def forward(self, x, steps, alpha):
        out = self.initial(x)
        
        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](upscaled)
            
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        
        return self.fade_in(alpha, final_upscaled, final_out)
        

class Discriminator(nn.Module):
    pass