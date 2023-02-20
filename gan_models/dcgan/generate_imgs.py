import wandb
import torch
from model_torch import Generator, Discriminator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first conv layer, default=64')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first conv layer, default=64')
parser.add_argument('gen_image', type=int, help='number of output images')


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize generator and discriminator
gen = Generator(args.nz, args.nc, args.ngf).to(device)
disc = Discriminator(args.nc, args.ndf, args.out_size).to(device)

gen.load_state_dict(wandb.load('gen.pth'))
disc.load_state_dict(wandb.load('disc.pth'))

#generate images
noise = torch.randn(args.gen_image, args.nz, 1, 1, device=device)
gen_imgs = gen(noise)
wandb.log({"generated_images": [wandb.Image(img, caption="Generated Image") for img in gen_imgs]})