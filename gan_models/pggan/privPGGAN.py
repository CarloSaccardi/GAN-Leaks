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
from torch.utils.data import Subset 

from model_torch import Generator, Discriminator, PrivateDiscriminator
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

parser.add_argument("--PATH", type=str, default=os.path.join(os.getcwd(), 'ersecki-thesis','model_save', 'privPGGAN'), help="Directory to save model")
parser.add_argument("--PATH_syn_data", type=str, default=os.path.join(os.getcwd(), 'ersecki-thesis', 'syn_data', 'privPGGA'), help="Directory to save synthetic data")

parser.add_argument("--save_model", type=bool, default=True, help="Save model or not")
parser.add_argument("--saved_model_name", type=str, default=None, help="Saved model name")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")

parser.add_argument('--ailab', type=bool, default=False, help='ailab server or not')

parser.add_argument("--N_splits", type=bool, default=2, help="number of gen-disc pairs")
parser.add_argument('--privacy_ratio', type=float, default=0.5, help='privacy ratio')
parser.add_argument('--disc_epochs', type=int, default=2, help='number of pretraining epochs for privateDisc')
parser.add_argument('--dp_delay', type=int, default=100, help='starting epoch for private discriminator')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True #improves the speed of the model

args = parser.parse_args()    

def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  


#args.local_config='gan_models/pggan/pggan_config.yaml'
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
pretrain_disc_epochs = [args.disc_epochs]*len(args.batch_size)

def get_loader(imge_size):
    transform = transforms.Compose([
        transforms.Resize((imge_size,imge_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5]*args.nc, [0.5]*args.nc)
    ])
    
    batch_size = args.batch_size[int(log2(imge_size) / 4)]
    dataset = CustomDataset(root=  args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    ####################### SPLIT DATASET ############################
    X = []
    t = len(dataset)//args.N_splits
    y_train = []
    X_loaders = []

    for i in range(args.N_splits):
        if i<args.N_splits-1:
            dataset_split = Subset(dataset, range(i*t, (i+1)*t))
            X_loaders += [torch.utils.data.DataLoader(dataset_split, batch_size=args.batch_size[0], shuffle=False)]
            X += [dataset_split]
            y_train += [i]*t
        else:
            dataset_split = Subset(dataset, range(i*t, len(dataset)))
            X += [dataset_split]
            X_loaders += [torch.utils.data.DataLoader(dataset_split, batch_size=args.batch_size[0], shuffle=False)]
            y_train += [i]*(len(dataset)-i*t)

    y_train = np.array(y_train) + 0.0 

    return X_loaders, X, y_train, loader


def train_fn_pretrain(private_critic, step_pretrain, alpha, opt_private_critic, loss_fn, loader, y_train):
    
    loop = tqdm(loader, leave=True)
    for i, (real) in enumerate(loop):
        
        real = real.to(device)
        cur_batch_size = real.shape[0]
        lables = y_train[i*cur_batch_size:(i+1)*cur_batch_size]
        lables = torch.tensor(lables).type(torch.LongTensor).to(device)
                    
        opt_private_critic.zero_grad()
        output = private_critic(real, step_pretrain, alpha)

        loss_critic = loss_fn(output, lables)
        loss_critic.backward()

        opt_private_critic.step()
         
        #train generator--> max E[critic(gen_fake)] --> min -E[critic(gen_fake)]
        
        alpha += cur_batch_size / ((pretrain_disc_epochs[step_pretrain] * 0.5) * len(loader.dataset)) 
        alpha = min(alpha, 1)
        
    return loss_critic, alpha


def train_fn(critic, 
             gen, 
             private_critic, 
             step, alpha, 
             opt_critic, 
             opt_gen, 
             opt_private_critic, 
             scaler_critic, 
             scaler_gen, 
             loss_fn, 
             loader, 
             split, 
             gen_y):
    
    loop = tqdm(loader, leave=True)
    for i, (real) in enumerate(loop):
        
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

        #train private discriminator if resolution is at least 32
        if 4*2**step >= args.dp_delay:
            output = private_critic(fake, step, alpha).reshape(-1, args.N_splits)
            labels_Dp = torch.tensor([split]*real.shape[0], dtype=torch.float32).type(torch.LongTensor).to(device)

            loss_Dp = loss_fn(output, labels_Dp)

            private_critic.zero_grad()
            loss_Dp.backward()

            opt_private_critic.step()
         
        #train generator--> max E[critic(gen_fake)] --> min -E[critic(gen_fake)]

        with torch.cuda.amp.autocast():
            output1 = critic(fake, step, alpha)
            output2 = private_critic(fake, step, alpha).reshape(-1, args.N_splits)
            labels_gen = gen_y[i*real.shape[0]:(i+1)*real.shape[0]]

            lossG1 = -torch.mean(output1)
            lossG2 = -args.privacy_ratio * loss_fn(output2, labels_gen.type(torch.LongTensor).to(device))
            loss_gen = lossG1 + lossG2

           
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

    print(args)

    if args.training:

        gen = Generator(args.nz, args.in_channels, args.nc).to(device)
        critic = Discriminator(args.in_channels, args.nc).to(device)
        private_critic = PrivateDiscriminator(args.in_channels, args.N_splits, args.nc).to(device)
        
        #initilize optimizers and scalers for FP16 training
        opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
        opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.99))
        opt_private_critic = torch.optim.Adam(private_critic.parameters(), lr=args.lr, betas=(0.0, 0.99))
        scaler_gen = torch.cuda.amp.GradScaler()
        scaler_critic = torch.cuda.amp.GradScaler()
        loss_fn = torch.nn.CrossEntropyLoss()
        
        if args.training:
            ####################### PRETRAIN DISCRIMINATOR ############################
            private_critic.train()
            step_pretrain = int(log2(args.start_img_size / 4))

            for num_epochs in pretrain_disc_epochs[step_pretrain:]:
                alpha = 1e-5 #increase to 1 over the course of the epoch
                _, _, y_train, loader = get_loader(4*2**step_pretrain)
                print(f"image resolution: {4*2**step_pretrain}x{4*2**step_pretrain}")

                for epoch in range(num_epochs):

                    loss_CRITIC,  alpha = train_fn_pretrain(private_critic, 
                                                            step_pretrain, 
                                                            alpha, 
                                                            opt_private_critic,  
                                                            loss_fn, 
                                                            loader,
                                                            y_train)

                    #print the losses for every epoch
                    print("pre-train private discriminator loss: ", loss_CRITIC.item(), "epoch: ", epoch)

                
                step_pretrain += 1
            ############################################################################

            ####################### TRAIN privPGGAN ############################

            gen.train()
            critic.train()
            step = int(log2(args.start_img_size / 4))
            genS = [gen]*args.N_splits
            criticS = [critic]*args.N_splits

            for num_epochs in PROGRESSIVE_EPOCHS[step:]:
                alpha = 1e-5 #increase to 1 over the course of the epoch
                X_loaders, X, y_train, loader = get_loader(4*2**step)
                print(f"image resolution: {4*2**step}x{4*2**step}")

                for epoch in range(num_epochs):

                    #for each dataloader in dataloader_N, train the discriminator
                    for split in range(args.N_splits):

                        l = list(range(args.N_splits))
                        del(l[split])
                        gen_y =  np.random.choice( l, ( len( X[split] ) , ) )
                        gen_y = torch.tensor(gen_y, dtype=torch.float32).to(device)

                        loss_CRITIC, loss_GEN, alpha = train_fn(
                                                                criticS[split], 
                                                                genS[split],
                                                                private_critic, 
                                                                step, 
                                                                alpha, 
                                                                opt_critic, 
                                                                opt_gen, 
                                                                opt_private_critic,
                                                                scaler_critic, 
                                                                scaler_gen, 
                                                                loss_fn,
                                                                X_loaders[split],
                                                                split,
                                                                gen_y
                                                                )
                        ##print losses, epoch and split
                        print("discriminator loss: ", loss_CRITIC.item(), "generator loss: ", loss_GEN.item(), "epoch: ", epoch)

                    
                step += 1
            ############################################################################

        if args.save_model:

            dirname = os.path.join(args.PATH, timestamp)

            os.makedirs(dirname, exist_ok=True)
            for i in range(args.N_splits):
                torch.save(genS[i].state_dict(), os.path.join(dirname, f"gen{i}.pth") )
                torch.save(criticS[i].state_dict(), os.path.join(dirname, f"critic{i}.pth") )
            torch.save(private_critic.state_dict(), os.path.join(dirname, "private_critic.pth"))

    if args.generate:
        #load the saved model, generate args.batch_size synthetic data, and save them as .npz file

        gen = Generator(args.nz, args.in_channels, args.nc).to(device)

        if args.training:
            gen.load_state_dict(torch.load(os.path.join(dirname, "gen0.pth")))
        else:
            assert args.saved_model_name is not None, "Please specify the saved model name"
            assert args.wandb == None, "No need to load anything to wand when only generating synthetic data"
            gen.load_state_dict(torch.load(os.path.join(args.saved_model_name, "gen0.pth")))
        
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
                normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
                to_pil = transforms.ToPILImage()

                batch_fake = gen(batch_noise, 4, 1).detach().cpu()
                batch_fake = normalize(batch_fake)
               
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