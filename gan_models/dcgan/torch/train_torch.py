from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model_torch import Generator, Discriminator, initialize_weights


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=int, default=128, help='batch size, default=128')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network, default=64')
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first conv layer, default=64')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first conv layer, default=64')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs, default=5')
parser.add_argument('dataroot', type=str, help='path to dataset')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

dataset = dset.ImageFolder(
    root=args.dataroot, 
    transforms = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(args.nc)], [0.5 for _ in range(args.nc)]),
                    ])
    )  


dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                           





