from __future__ import print_function
#%matplotlib inline
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np
import numpy as np
import wandb
import yaml
import warnings

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
parser.add_argument('out_size', type=int, help='number of output images')
parser.add_argument('beta1', type=float, help='beta1 for adam. default=0.5')
parser.add_argument('beta2', type=float, help='beta2 for adam. default=0.999')
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
    disc = Discriminator(args.nc, args.ndf, args.out_size).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCELoss()

    gen.train()
    disc.train()

    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # train discriminator --> max log(D(x)) + log(1 - D(G(z)))
            disc.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), 1, device=device)
            output = disc(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(0)
            output = disc(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            opt_disc.step()

            # train generator --> min log(1-D(G(z))) <-> max log(D(G(z)))
            gen.zero_grad()
            label.fill_(1)
            output = disc(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            opt_gen.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, args.num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        if args.wandb:
            
            #log metrics to wandb
            wandb.log({"epoch": epoch, "loss_disc": errD.item(), "loss_gen": errG.item(), "D(x)": D_x, "D(G(z))": D_G_z1, "D(G(z))": D_G_z2})

            #visualize progress after each epoch in wandb
            with torch.no_grad():
                fake = gen(noise).detach().cpu()
                fake = (fake + 1) / 2  # scale the values to be between 0 and 1
                grid = np.vstack([np.hstack([fake[i + j * 4] for i in range(4)]) for j in range(4)])
                wandb.log({"examples": wandb.Image(grid, caption="epoch: {}".format(epoch))})
       
    #save model parameters
    wandb.save(gen.state_dict(), 'gen.pth')
    wandb.save(disc.state_dict(), 'disc.pth')
  
    
    
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
   
        


                          





