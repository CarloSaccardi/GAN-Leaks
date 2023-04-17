import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log2
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
import argparse
import os
import wandb
import warnings
import yaml
import datetime
from torch.utils.data import Subset

from model_torch import Generator, Discriminator
from utils import gradient_penalty, CustomDataset


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=list, default=[16, 16, 16, 16], help='batch size for each resolution')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network, default=64')
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
parser.add_argument('--in_channels', type=int, default=512, help='number of generator filters in first conv layer, default=256')
parser.add_argument('--start_img_size', type=int, default=4, help='starting image size, default=4')
parser.add_argument('--lambda_gp', type=float, default=10, help='lambda for gradient penalty')
parser.add_argument('--data_name', type=str, default='miniCelebA', help='name of the dataset, either miniCelebA or CelebA')
parser.add_argument('--local_config', default=None, help='path to config file')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")

parser.add_argument("--PATH", type=str, default=os.path.join(os.getcwd(),'model_save', 'pggan'), help="Directory to save model")
parser.add_argument("--PATH_syn_data", type=str, default=os.path.join(os.getcwd(), 'syn_data', 'pggan'), help="Directory to save synthetic data")

parser.add_argument("--save_model", type=bool, default=True, help="Save model or not")
parser.add_argument("--saved_model_name", type=str, default=None, help="Saved model name")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True #improves the speed of the model

args = parser.parse_args()
PROGRESSIVE_EPOCHS = [20]*len(args.batch_size)
FIXED_NOISE = torch.randn(8, args.nz, 1, 1).to(device)


def get_loader(imge_size):
    transform = transforms.Compose([
        transforms.Resize((imge_size,imge_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5]*args.nc, [0.5]*args.nc)
    ])
    
    batch_size = args.batch_size[int(log2(imge_size) / 4)]
    dataset = CustomDataset(root= os.path.join('data', args.data_name), transform=transform, n = 10000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    ####################### SPLIT DATASET ############################
    X = []
    t = len(dataset)//args.N_splits
    y_train = []
    X_loaders = []

    for i in range(args.N_splits):
        if i<args.N_splits-1:
            dataset_split = Subset(dataset, range(i*t, (i+1)*t))
            X_loaders += [torch.utils.data.DataLoader(dataset_split, batch_size=args.batch_size, shuffle=False)]
            X += [dataset_split]
            y_train += [i]*t
        else:
            dataset_split = Subset(dataset, range(i*t, len(dataset)))
            X += [dataset_split]
            X_loaders += [torch.utils.data.DataLoader(dataset_split, batch_size=args.batch_size, shuffle=False)]
            y_train += [i]*(len(dataset)-i*t)

    y_train = np.array(y_train) + 0.0 

    return X_loaders, X, y_train, loader


def main():
    #TODO: add seed for reproducibility and initialization of weights


    now = datetime.datetime.now() # To create a unique folder for each run
    timestamp = now.strftime("_%Y_%m_%d__%H_%M_%S")  # To create a unique folder for each run

    print(args)

    if args.training:

        gen = Generator(args.nz, args.in_channels, args.nc).to(device)
        critic = Discriminator(args.in_channels, args.nc).to(device)
        
        #initilize optimizers and scalers for FP16 training
        opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
        opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.99))
        scaler_gen = torch.cuda.amp.GradScaler()
        scaler_critic = torch.cuda.amp.GradScaler()
        
        gen.train()
        critic.train()
        step = int(log2(args.start_img_size / 4))