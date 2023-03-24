import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import Subset, TensorDataset, ConcatDataset
import numpy as np
import numpy as np
import wandb
import yaml
import warnings
import datetime
import torchvision

from model_torch import Generator, Discriminator, PrivateDiscriminator, initialize_weights

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
parser.add_argument('--disc_epochs', type=int, default=2, help='number of pretraining epochs for privateDisc')
parser.add_argument('--dp_delay', type=int, default=100, help='starting epoch for private discriminator')
parser.add_argument('--out_size', type=int, help='number of output images')
parser.add_argument('--beta1', type=float, default=0.5 , help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--privacy_ratio', type=float, default=0.5, help='privacy ratio')
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

#### PRIVGAN PARAMETERS ####
parser.add_argument("--N_splits", type=bool, default=2, help="number of gen-disc pairs")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()


transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(args.nc)], [0.5 for _ in range(args.nc)])
                ])


dataset = dset.ImageFolder(root= os.path.join('data', args.data_name), transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


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

fixed_noise = torch.randn(1, args.nz, 1, 1, device=device)

def main():
    #TODO: add seed for reproducibility and initialization of weights
    print(args)
    
    now = datetime.datetime.now() # To create a unique folder for each run
    timestamp = now.strftime("_%Y_%m_%d__%H_%M_%S")  # To create a unique folder for each run
    
    gen = Generator(args.nz, args.nc, args.ngf).to(device)
    disc = Discriminator(args.nc, args.ndf).to(device)
    private_disc = PrivateDiscriminator(args.nc, args.ndf).to(device)
    initialize_weights(gen)
    initialize_weights(disc)
    initialize_weights(private_disc)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_private_disc = optim.Adam(private_disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    loss_fn = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()

    gen.train()
    disc.train()
    private_disc.train()

    #pre-train private discriminator for disc_epochs. Use Y_train as labels
    for epoch in range(args.disc_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            lables = y_train[i*batch_size:(i+1)*batch_size]
            lables = torch.tensor(lables).to(device)

            opt_private_disc.zero_grad()
            output = private_disc(imgs).reshape(-1)

            loss = loss_fn(output, lables)
            loss.backward()

            opt_private_disc.step()


    train_privGAN(genS = [gen]*args.N_splits ,
                   discS = [disc]*args.N_splits, 
                   criterion = criterion,
                   opt_gen = opt_gen, 
                   opt_disc = opt_disc, 
                   private_disc=private_disc,
                   opt_private_disc = opt_private_disc,
                   t = t, 
                   loss_fn = loss_fn)


def train_privGAN(genS, discS, private_disc, opt_gen, opt_disc, opt_private_disc, t, criterion, loss_fn):
    #train generator and discriminator for gen_epochs
    batchCount = int(t // args.batch_size)

    for epoch in range(args.num_epochs):

        d_t = np.zeros((args.N_splits, batchCount))
        dp_t = np.zeros(batchCount)
        g_t = np.zeros(batchCount)
        privD_dataset = []
        gen_dataset = []

        #for each dataloader in dataloader_N, train the discriminator
        for split in range(args.N_splits):

            imgsF_split = []

            for _, (imgs, _) in enumerate(X_loaders[split]):

                imgs = imgs.to(device)
                noise = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
                fake = genS[split](noise)
                imgsF_split += [fake.detach().numpy()]

                # train discriminator --> max log(D(x)) + log(1 - D(G(z)))
                disc_real = discS[split](imgs).reshape(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = discS[split](fake.detach()).reshape(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                discS[split].zero_grad()
                lossD.backward()
                opt_disc.step()

                d_t[split, i] = lossD.item()

            #prepare labels for generator: random numbers as the goal is not only too fool 
            #the discriminator but also the private discriminator. A random lable makes the
            #generator generate images that are not too similar to the ones from the same split


            X_fakes_split = torch.tensor(np.concatenate(imgsF_split, axis=0), dtype=torch.float32)
            l = list(range(args.N_splits))
            del(l[split])
            gen_y =  np.random.choice( l, ( len( X[split] ) , ) ) 
            gen_y = torch.tensor(gen_y, dtype=torch.float32).to(device)
            gen_dataset.append(TensorDataset(X_fakes_split, gen_y))

            if epoch > args.dp_delay:
                privD_y = torch.tensor([split]*len(X[split]), dtype=torch.float32).to(device)
                privD_dataset.append(TensorDataset(X_fakes_split, privD_y))

        if epoch > args.dp_delay:

            #concatenate all the data from all the splits
            imgsF_dataset = ConcatDataset(privD_dataset)
            imgsF_dataloader = torch.utils.data.DataLoader(imgsF_dataset, batch_size=args.batch_size, shuffle=True)

            #train private discriminator with yDisc2 as labels
            for i, (imgs, lables) in enumerate(imgsF_dataloader):

                imgs = imgs.to(device)
                lables = lables.to(device)
                output = private_disc(imgs).reshape(-1)
                lossD_private = loss_fn(output, lables)

                private_disc.zero_grad()
                lossD_private.backward()
                
                opt_private_disc.step()

                dp_t[i] = lossD_private.item()

        #train generators: goal is to fool the discriminator and the private discriminator
        for split in range(args.N_splits):

            gen_dataset_train = ConcatDataset(gen_dataset)
            gen_dataset_train = torch.utils.data.DataLoader(gen_dataset_train, batch_size=args.batch_size, shuffle=True)

            for i, (imgs, lables) in enumerate(gen_dataset_train):

                imgs = imgs.to(device)#this is the fake image
                output1 = discS[split](imgs).reshape(-1)
                output2 = private_disc(imgs).reshape(-1)

                lossG_1 = criterion(output1, torch.ones_like(output1))
                lossG_2 = args.privacy_ratio * loss_fn(output2, lables)

                lossG = lossG_1 + lossG_2

                genS[split].zero_grad()
                lossG.backward()
                opt_gen.step()
            
                g_t[i] = lossG.item()
        


                




if __name__ == '__main__':
    main()