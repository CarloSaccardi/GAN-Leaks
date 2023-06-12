import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):   #weight scaled conv layer, EQULIZED LEARNING RATE
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2): #gain is for the initialization constant
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5 
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
        super(PixelNorm, self).__init__()
        self.eps = 1e-8
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

class ConvBlock(nn.Module): # 3x3 conv block with pixel norm and leaky relu
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixel_norm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        
    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
        
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),# 1x1 conv transpose to 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
        
        self.initial_rgb =  WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        
        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]), nn.ModuleList([self.initial_rgb]))
        
        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WSConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))
        
        
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
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        
        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))
        
        #this is for he 4x4 img resolution
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        #block for the 4x4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0,  stride=1),
        )
        
        
        
    def fade_in(self, alpha, downscaled, real):
        return alpha * real + (1 - alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1)
    
    def forward(self, x, steps, alpha):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))
        
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
            
        out = self.minibatch_std(out)
        return self.final_block(out).view(x.shape[0], -1)
    




class PrivateDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, img_channels=3):
        super(PrivateDiscriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        
        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))
        
        #this is for he 4x4 img resolution
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        #block for the 4x4 resolution
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, out_channels, kernel_size=1, padding=0,  stride=1),
            nn.Softmax()
        )
        
        
        
    def fade_in(self, alpha, downscaled, real):
        return alpha * real + (1 - alpha) * downscaled
    
    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1)
    
    def forward(self, x, steps, alpha):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))
        
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
            
        out = self.minibatch_std(out)
        return self.final_block(out).view(x.shape[0], -1)
    

class stackGenerators(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels, num_generators):
        super(stackGenerators, self).__init__()
        self.num_generators = num_generators
        self.gen = nn.ModuleList()
        for i in range(self.num_generators):
            self.gen.append(Generator(z_dim, in_channels, img_channels))

    def forward(self, x, steps, alpha, i):
        return self.gen[i](x, steps, alpha)
    
class stackDiscriminators(nn.Module):
    def __init__(self, in_channels, img_channels, num_discriminators):
        super(stackDiscriminators, self).__init__()
        self.num_discriminators = num_discriminators
        self.disc = nn.ModuleList()
        for i in range(self.num_discriminators):
            self.disc.append(Discriminator(in_channels, img_channels))

    def forward(self, x, steps, alpha, i):
        return self.disc[i](x, steps, alpha)
    
    
if __name__ == '__main__':
    z_dim = 512
    in_channels = 512
    img_channels = 3
    batch_size = 16
    steps = 2
    alpha = 0.5
    
    g = Generator(z_dim, in_channels, img_channels)
    d = Discriminator(z_dim, in_channels, img_channels)
    
    z = torch.randn(batch_size, z_dim, 1, 1)
    x = torch.randn(batch_size, img_channels, 2**steps * 4, 2**steps * 4)
    
    print(g(z, steps, alpha).shape)
    print(d(x, steps, alpha).shape)
        
            
            
            
    
    