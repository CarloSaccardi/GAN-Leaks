import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from math import log2
from tqdm import tqdm
import argparse

from model_torch import Generator, Discriminator
from utils import gradient_penalty


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=list, default=[16, 16, 16, 16, 16, 16, 16, 8, 4], help='batch size for each resolution')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network, default=64')
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
parser.add_argument('--in_channels', type=int, default=512, help='number of generator filters in first conv layer, default=256')
parser.add_argument('--start_img_size', type=int, default=4, help='starting image size, default=4')
parser.add_argument('lambda_gp', type=float, default=10, help='lambda for gradient penalty')

torch.backends.cudnn.benchmark = True #improves the speed of the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
PROGRESSIVE_EPOCHS = [20]*len(args.batch_size)
FIXED_NOISE = torch.randn(8, args.nz, 1, 1).to(device)


def get_loader(imge_size):
    transform = transforms.Compose([
        transforms.Resize(imge_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5]*args.nc, [0.5]*args.nc)
    ])
    
    batch_size = args.batch_size[int(log2(imge_size)) / 4]
    
    dataset = datasets.ImageFolder(root='data', transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader, dataset
    

def train_fn(critic, gen, step, alpha, opt_critic, opt_gen, scaler_critic, scaler_gen, loader):
    
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        
        real = real.to(device)
        cur_batch_size = real.shape[0]
         
        #train critic--> max ( E[critic(real)] - E[critic(fake)] )
        noise = torch.randn(cur_batch_size, args.nz, 1, 1).to(device)
        
        with torch.cuda.amp.autocast():
            fake = gen(noise, step, alpha)
            critic_real = critic(real,  step, alpha)
            critic_fake = critic(fake.detach(), step, alpha)
            
            gp = gradient_penalty(critic, real, fake, step, alpha)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) 
                + args.lambda_gp * gp
                + 0.001 * torch.mean(critic_real ** 2)
               )
             
        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward() 
        scaler_critic.step(opt_critic)
        scaler_critic.update()
         
        #train generator--> max E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, step, alpha)
            loss_gen = -torch.mean(gen_fake)
           
        opt_critic.zero_grad()
        scaler_critic.scale(loss_gen).backward()
        scaler_critic.step(opt_critic)
        scaler_gen.update()
        
        alpha += cur_batch_size / (len(loader.dataset) * PROGRESSIVE_EPOCHS[step] * 0.5)
        alpha = min(alpha, 1)
        
    return loss_critic, loss_gen, alpha
        

def main():
    gen = Generator(args.nz, args.in_channels, args.nc).to(device)
    critic = Discriminator(args.in_channels, args.nc).to(device)
    
    #initilize optimizers and scalers for FP16 training
    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.99))
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()
    
    #TODO: add a function to get the loader and wandb
    
    gen.train()
    critic.train()
    step = int(log2(args.start_img_size / 4))
    
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        print(f"iamge resolution: {4*2**step}x{4*2**step}")
        
        loss_critic, loss_gen, alpha = train_fn(critic, 
                                                gen, 
                                                step, 
                                                alpha, 
                                                opt_critic, 
                                                opt_gen, 
                                                scaler_critic, 
                                                scaler_gen, 
                                                loader)
        
        #TODO: add wandb logging and save the model
        
    step += 1
        
        

if __name__ == '__main__':
    main()

