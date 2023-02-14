from __future__ import print_function
#%matplotlib inline
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np
import numpy as np
import wandb
import yaml
import warnings

from model import Generator, Discriminator, initialize_weights
from utils import grandient_penalty


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=int, default=64, help='batch size, default=128')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network, default=64')
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first conv layer, default=64')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first conv layer, default=64')
parser.add_argument('critic_iter', type=int, default=5, help='number of training iteration for the critic')
parser.add_argument('lambda_gp', type=float, default=10, help='gradient penalty lambda hyperparameter')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs, default=5')
parser.add_argument('out_size', type=int, help='number of output images')
parser.add_argument('beta1', type=float, help='beta1 for adam. default=0.0')
parser.add_argument('beta2', type=float, help='beta2 for adam. default=0.9')
parser.add_argument('dataroot', type=str, default=1, help='path to dataset')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")

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

def main():
    
    gen = Generator(args.nz, args.nc, args.ngf).to(device)
    critic = Discriminator(args.nc, args.ndf, args.out_size).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas= (args.beta1, args.beta2))
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr, betas= (args.beta1, args.beta2))

    gen.train()
    critic.train()

    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # train discriminator --> max log(D(x)) + log(1 - D(G(z)))
            real = data[0].to(device)

            for _ in range(args.critic_iter):
                noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
                fake = gen(noise)
                gp = grandient_penalty(critic, real, fake, device=device)
                loss = critic(fake).mean() - critic(real).mean() + args.lambda_gp * gp
                critic.zero_grad()
                loss.backward(retain_graph=True)
                opt_critic.step()

            # train generator --> min -E[critic(gen_fake)]
            output = critic(fake).mean()
            loss_gen = -output
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # print statistics and log to wandb
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, args.num_epochs, i, len(dataloader),
                         loss.item(), loss_gen.item()))
                if args.wandb:
                    wandb.log({"Loss_D": loss.item(), "Loss_G": loss_gen.item()})
                    # visualize progress in wandb
                    noise = torch.randn(64, args.nz, 1, 1, device=device)
                    fake = gen(noise).detach().cpu()
                    grid = torchvision.utils.make_grid(fake)
                    wandb.log({"generated_images": [wandb.Image(grid, caption="Epoch {}".format(epoch))]})


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)    
    
if __name__ == '__main__':
    
    if args.local_config is not None:
        with open(str(args.local_config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)
        if args.wandb:
            wandb_config = vars(args)
            run = wandb.init(project=str(args.wandb), entity="THESIS", config=wandb_config)
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")
    main(args)
   
        


                          





