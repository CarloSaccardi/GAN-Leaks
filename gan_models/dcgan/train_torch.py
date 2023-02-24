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
import datetime
import torchvision

from model_torch import Generator, Discriminator, initialize_weights

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default=0.0002')
parser.add_argument('--batch_size', type=int, default=128, help='batch size, default=128')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network, default=64')
parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input image, default=3')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector, default=100')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in first conv layer, default=64')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters in first conv layer, default=64')
parser.add_argument('--input_size', type=int, default=64, help='size of the input image to network, default=64')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs, default=5')
parser.add_argument('--out_size', type=int, help='number of output images')
parser.add_argument('--beta1', type=float, default=0.5 , help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--data_name', type=str, default='miniCelebA', help='name of the dataset, either miniCelebA or CelebA')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")
parser.add_argument('--local_config', default=None, help='path to config file')

parser.add_argument("--PATH", type=str, default=os.path.join(os.getcwd(),'model_save', 'dcgan'), help="Directory to save model")
parser.add_argument("--PATH_syn_data", type=str, default=os.path.join(os.getcwd(), 'syn_data', 'dcgan'), help="Directory to save synthetic data")

parser.add_argument("--save_model", type=bool, default=True, help="Save model or not")
parser.add_argument("--saved_model_name", type=str, default=None, help="Saved model name")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(args.nc)], [0.5 for _ in range(args.nc)])
                ])


dataset = dset.ImageFolder(root= os.path.join('data', args.data_name), transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

fixed_noise = torch.randn(1, args.nz, 1, 1, device=device)

def main():
    #TODO: add seed for reproducibility and initialization of weights
    print(args)
    
    now = datetime.datetime.now() # To create a unique folder for each run
    timestamp = now.strftime("_%Y_%m_%d__%H_%M_%S")  # To create a unique folder for each run

    if args.training:
    
        gen = Generator(args.nz, args.nc, args.ngf).to(device)
        disc = Discriminator(args.nc, args.ndf).to(device)
        initialize_weights(gen)
        initialize_weights(disc)

        opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        criterion = nn.BCELoss()

        gen.train()
        disc.train()

        for epoch in range(args.num_epochs):
            for batch_idx, (real, _) in enumerate(dataloader):
                
                real = real.to(device)
                noise = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
                fake = gen(noise)

                # train discriminator --> max log(D(x)) + log(1 - D(G(z)))
                disc_real = disc(real).reshape(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake.detach()).reshape(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                disc.zero_grad()
                lossD.backward()
                opt_disc.step()

                # train generator --> min log(1-D(G(z))) <-> max log(D(G(z)))
                output = disc(fake).reshape(-1)
                lossG = criterion(output, torch.ones_like(output))
                gen.zero_grad()
                lossG.backward()
                opt_gen.step()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch [{epoch}/{args.num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                            Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                    )
            
            if args.wandb:
                
                #log metrics to wandb
                wandb.log({"epoch": epoch, "loss_disc": lossD.item(), "loss_gen": lossG.item()})

                #visualize progress after each epoch in wandb
                with torch.no_grad():   
                    fake_wb = gen(fixed_noise).detach().cpu()#shape (1, 3, 64, 64)
                    grid = torchvision.utils.make_grid(fake_wb[0], normalize=True)
                    wandb.log({"generated_images": wandb.Image(grid, caption="epoch: {}".format(epoch))})
        

            #save model parameters
        if args.save_model:
            dirname = os.path.join(args.PATH, timestamp)
            os.makedirs(dirname, exist_ok=True)
            torch.save(gen.state_dict(), os.path.join(dirname, "generator.pth"))
            torch.save(disc.state_dict(), os.path.join(dirname, "discriminator.pth"))

    if args.generate:
        #load the saved model, generate args.batch_size synthetic data, and save them as .npz file

        gen = Generator(args.nz, args.nc, args.ngf).to(device)

        if args.training:
            gen.load_state_dict(torch.load(os.path.join(dirname, "generator.pth")))
        else:
            assert args.saved_model_name is not None, "Please specify the saved model name" 
            assert args.wandb == None, "No need to load anything to wand when only generating synthetic data"
            gen.load_state_dict(torch.load(os.path.join(args.saved_model_name, "generator.pth")))
        
        gen.eval()

        with torch.no_grad():
            noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
            normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
            to_pil = transforms.ToPILImage()

            fake = gen(noise).detach().cpu()
            fake = normalize(fake)

            dirname = os.path.join(args.PATH_syn_data , 'npz_images', timestamp)
            os.makedirs(dirname, exist_ok=True)
            np.savez(os.path.join(dirname, "dcgan_synthetic_data.npz"), fake=fake)

            dirname = os.path.join(args.PATH_syn_data , 'npz_noise', timestamp)
            os.makedirs(dirname, exist_ok=True)
            np.savez(os.path.join(dirname, "dcgan_noise.npz"), noise=noise)

            dirname = os.path.join(args.PATH_syn_data , 'png_images', timestamp)
            os.makedirs(dirname, exist_ok=True)
            for i, img in enumerate(fake):
                pil_img = to_pil(img)
                save_path = os.path.join(dirname, f"image_{i}.png")
                pil_img.save(save_path)

    
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
            run = wandb.init(project=str(args.wandb), entity="thesis_carlo", config=wandb_config)
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")

    main()