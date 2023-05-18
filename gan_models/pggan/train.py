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
import warnings
import yaml
import datetime
import math
from PIL import Image

from model_torch import Generator, Discriminator
from utils import gradient_penalty, CustomDataset


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=30, help='number of training epochs for each resolution')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=list, default=[16, 16, 16, 16, 16], help='batch size for each resolution')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network, default=64')
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector, default=100')
parser.add_argument('--in_channels', type=int, default=256, help='number of generator filters in first conv layer, default=256')
parser.add_argument('--start_img_size', type=int, default=4, help='starting image size, default=4')
parser.add_argument('--num_generated', type=int, default=10000, help='number of generated images')
parser.add_argument('--lambda_gp', type=float, default=10, help='lambda for gradient penalty')
parser.add_argument('--data_path', type=str, default='miniCelebA', help='name of the dataset, either miniCelebA or CelebA')
parser.add_argument('--local_config', default=None, help='path to config file')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")

parser.add_argument("--PATH", type=str, default=os.path.join(os.getcwd(), 'ersecki-thesis','model_save', 'dcgan'), help="Directory to save model")
parser.add_argument("--PATH_syn_data", type=str, default=os.path.join(os.getcwd(), 'ersecki-thesis', 'syn_data', 'dcgan'), help="Directory to save synthetic data")

parser.add_argument("--save_model", type=bool, default=True, help="Save model or not")
parser.add_argument("--saved_model_name", type=str, default=None, help="Saved model name")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")

parser.add_argument('--ailab', type=bool, default=False, help='ailab server or not')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True #improves the speed of the model

args = parser.parse_args()    

def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  

if args.local_config is not None:
    with open(str(args.local_config), "r") as f:
        config = yaml.safe_load(f)
    update_args(args, config)

    if not args.ailab:
        import wandb #wandb is not supported on ailab server

    if args.wandb:
        wandb_config = vars(args)
        run = wandb.init(project=str(args.wandb), entity="thesis_carlo", config=wandb_config)
        # update_args(args, dict(run.config))
else:
    warnings.warn("No config file was provided. Using default parameters.")

PROGRESSIVE_EPOCHS = [args.num_epochs]*len(args.batch_size)
print(args)

def get_loader(imge_size):
    transform = transforms.Compose([
        transforms.Resize(size=(imge_size,imge_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(args.nc)], 
                             [0.5 for _ in range(args.nc)])
    ])
    
    batch_size = args.batch_size[int(log2(imge_size) / 4)]
    dataset = CustomDataset(root=  args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader, dataset
    

def train_fn(critic, gen, step, alpha, opt_critic, opt_gen, scaler_critic, scaler_gen, loader):
    
    loop = tqdm(loader, leave=True)
    for _, (real) in enumerate(loop):
        
        real = real.to(device)
        cur_batch_size = real.shape[0]
         
        #train critic--> max ( E[critic(real)] - E[critic(fake)] ) --> min ( - E[critic(real)] + E[critic(fake)] )
        noise = torch.randn(cur_batch_size, args.nz, 1, 1).to(device)
        
        with torch.cuda.amp.autocast():
            fake = gen(noise, step, alpha)
            critic_real = critic(real,  step, alpha)
            critic_fake = critic(fake.detach(), step, alpha)
            
            gp = gradient_penalty(critic, real, fake, alpha, step, device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) 
                + args.lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
               )
             
        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward() #retain_graph=True to be able to backprop the generator
        scaler_critic.step(opt_critic)
        scaler_critic.update()
         
        #train generator--> max E[critic(gen_fake)] --> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, step, alpha)
            loss_gen = -torch.mean(gen_fake)
           
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        
        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(loader.dataset)) 
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )
        
    return loss_critic, loss_gen, alpha
        

def main():

    #TODO: add seed for reproducibility and initialization of weights

    now = datetime.datetime.now() # To create a unique folder for each run
    timestamp = now.strftime("_%Y_%m_%d__%H_%M_%S")  # To create a unique folder for each run

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
    
        for num_epochs in PROGRESSIVE_EPOCHS[step:]:
            alpha = 1e-5 #increase to 1 over the course of the epoch
            loader, _ = get_loader(4*2**step)
            print(f"image resolution: {4*2**step}x{4*2**step}")

            for epoch in range(num_epochs):

                loss_CRITIC, loss_GEN, alpha = train_fn(critic, 
                                                        gen, 
                                                        step, 
                                                        alpha, 
                                                        opt_critic, 
                                                        opt_gen, 
                                                        scaler_critic, 
                                                        scaler_gen, 
                                                        loader)
                
                #print the losses for every epoch
                print(f"epoch: {epoch}, loss_critic: {loss_CRITIC}, loss_gen: {loss_GEN}")

                #wandb logging and save the model
                if args.wandb:
                    #visualize progress after every epoch in wandb
                    with torch.no_grad():
                        wandb.log({"loss_critic": loss_CRITIC, "loss_gen": loss_GEN})
                        noise = torch.randn(1, args.nz, 1, 1, device=device)
                        fake = gen(noise, step, alpha)
                        grid = torchvision.utils.make_grid(fake, normalize=True)
                        wandb.log({"progress": [wandb.Image(grid, caption=f"step: {step}, alpha: {alpha}")]})
            
            step += 1

        if args.save_model:
            dirname = os.path.join(args.PATH, timestamp)
            os.makedirs(dirname, exist_ok=True)
            torch.save(gen.state_dict(), os.path.join(dirname, "generator.pth"))
            torch.save(critic.state_dict(), os.path.join(dirname, "critic.pth"))
            print("saved model")

    if args.generate:
        #load the saved model, generate args.batch_size synthetic data, and save them as .npz file

        gen = Generator(args.nz, args.in_channels, args.nc).to(device)

        if args.training:
            gen.load_state_dict(torch.load(os.path.join(dirname, "generator.pth")))
        else:
            assert args.saved_model_name is not None, "Please specify the saved model name"
            assert args.wandb == None, "No need to load anything to wand when only generating synthetic data"
            gen.load_state_dict(torch.load(os.path.join(args.saved_model_name, "generator.pth")))
        
        gen.eval()

        with torch.no_grad():
            batch_size = 32  # Specify the batch size for generating images

            num_batches = math.ceil(args.num_generated / batch_size)
            #create empty tensor for noise and fake images
            noise = torch.empty((args.num_generated, args.nz, 1, 1), device=device)
            fake = torch.empty((args.num_generated, args.nc, args.image_size, args.image_size), device=device)

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, args.num_generated)

                batch_noise = torch.randn(batch_end - batch_start, args.nz, 1, 1, device=device)
                #normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
                to_pil = transforms.ToPILImage()

                batch_fake = gen(batch_noise, 4, 1).detach().cpu() * 0.5 + 0.5
                #batch_fake = normalize(batch_fake)
               
                noise[batch_start:batch_end] = batch_noise
                fake[batch_start:batch_end] = batch_fake

                dirname = os.path.join(args.PATH_syn_data, 'png_images', timestamp)
                os.makedirs(dirname, exist_ok=True)
                for i, img in enumerate(batch_fake):
                    pil_img = to_pil(img)
                    save_path = os.path.join(dirname, f"image_{batch_start + i}.png")
                    pil_img.save(save_path)

            dirname = os.path.join(args.PATH_syn_data, 'npz_images', timestamp)
            os.makedirs(dirname, exist_ok=True) 
            np.savez(os.path.join(dirname, f"pggan_images.npz"), fake=fake.cpu())

            dirname = os.path.join(args.PATH_syn_data, 'npz_noise', timestamp)
            os.makedirs(dirname, exist_ok=True)
            np.savez(os.path.join(dirname, f"pggan_noise.npz"), noise=noise.cpu())


if __name__ == '__main__':
    main()