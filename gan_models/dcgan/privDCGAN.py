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
import yaml
import warnings
import datetime
import math
from utils import CustomDataset, MySubDataset
import itertools

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
parser.add_argument('--data_path', type=str, default='miniCelebA', help='name of the dataset, either miniCelebA or CelebA')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")
parser.add_argument('--local_config', default=None, help='path to config file')
parser.add_argument('--hyperparameter_search',  default=None, help='path to config file')
parser.add_argument('--num_images', type=int, default=1000, help='number of images to use for training')

parser.add_argument("--PATH", type=str, default=os.path.join(os.getcwd(), 'ersecki-thesis', 'model_save', 'privDCGAN'), help="Directory to save model")
parser.add_argument("--PATH_syn_data", type=str, default=os.path.join(os.getcwd(), 'ersecki-thesis', 'syn_data', 'privDCGAN'), help="Directory to save synthetic data")

parser.add_argument("--save_model", type=bool, default=True, help="Save model or not")
parser.add_argument("--saved_model_name", type=str, default=None, help="Saved model name")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
#### PRIVGAN PARAMETERS ####
parser.add_argument("--N_splits", type=bool, default=2, help="number of gen-disc pairs")
parser.add_argument('--privacy_ratio', type=float, default=0.5, help='privacy ratio')

parser.add_argument('--ailab', type=bool, default=False, help='ailab server or not')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

fixed_noise = torch.randn(1, args.nz, 1, 1, device=device)

torch.autograd.set_detect_anomaly(True)

def main():
    #set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0) 

    now = datetime.datetime.now() # To create a unique folder for each run
    timestamp = now.strftime("_%Y_%m_%d__%H_%M_%S")  # To create a unique folder for each run

    #### HYPERPARAMETER SEARCH, DEFINE PARAMETERS TO TUNE ####
    if args.hyperparameter_search is not None:
        with open(str(args.hyperparameter_search), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)
        keys, values = zip(*config.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]#make a list of possible combinations for hyperparameters 
        args.params_keys = '-'.join(keys)
    else:
        experiments = [{}]
        args.params_values = None
        args.params_keys = None
    ###############################################s

    for i in range(len(experiments)):

        if len(experiments)>1:#if we are doing hyperparameter search
            exp = experiments[i]#take the i-th combination of hyperparameters
            update_args(args, exp)#update args with the hyperparameters
            args.params_values = '-'.join([str(v) for v in exp.values()])#string of hyperparameters values 

        if args.wandb: 
            wandb_config = vars(args)
            wandb.init(project=str(args.wandb), entity="thesis_carlo", config=wandb_config) 
            

        print(args)

        transform = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5 for _ in range(args.nc)], [0.5 for _ in range(args.nc)])
                        ])
        dataset = CustomDataset(root= args.data_path, transform=transform)
        assert len(dataset) % args.N_splits == 0, "Dataset size must be divisible by N_splits"
        t = len(dataset)//args.N_splits
        lables = [[i]*t for i in range(args.N_splits)]
        lables = list(itertools.chain.from_iterable(lables))
        dataset.labels = torch.tensor(lables)
        X_loaders = []
        for i in range(args.N_splits):
            filtered_dataset = [(data, label) for data, label in dataset if label == i]
            filtered_dataset = MySubDataset(filtered_dataset)
            X_loaders += [torch.utils.data.DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=True)]

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


        ####################### SPLIT DATASET ############################
        """
        X = []
        
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
        """
        ###################################################################

        ####################### DEFINE MODELS ############################
        
        gen = Generator(args.nz, args.nc, args.ngf).to(device)
        disc = Discriminator(args.nc, args.ndf).to(device)
        private_disc = PrivateDiscriminator(args.nc, args.ndf, args.N_splits).to(device)
        #initialize weights for each gen disc pair
        initialize_weights(gen)
        initialize_weights(disc)
        opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) 
        opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) 
        opt_private_disc = optim.Adam(private_disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        loss_fn = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.BCELoss()
        ###################################################################

        if args.training:

            private_disc.train()
            gen.train()
            disc.train()

            #### PRE-TRAIN PRIVATE DISCRIMINATOR ####
            for epoch in range(args.disc_epochs):
                for i, (imgs, labels) in enumerate(dataloader):
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    opt_private_disc.zero_grad()
                    output = private_disc(imgs).reshape(-1, args.N_splits)
                    loss = loss_fn(output, labels)
                    loss.backward()

                    opt_private_disc.step()

                print("pre-train private discriminator loss: ", loss.item(), "epoch: ", epoch)
            ##########################################

            train_privGAN(genS = [gen]*args.N_splits ,
                    discS = [disc]*args.N_splits, 
                    criterion = criterion,
                    opt_gen = opt_gen, 
                    opt_disc = opt_disc, 
                    private_disc=private_disc,
                    opt_private_disc = opt_private_disc,
                    t = t, 
                    loss_fn = loss_fn,
                    X_loaders = X_loaders,
                    timestamp=timestamp)
            
            if args.wandb:
                wandb.finish()

        if args.generate:
            #load the saved model, generate args.batch_size synthetic data, and save them as .npz file

            gen = Generator(args.nz, args.nc, args.ngf).to(device)
            
            if args.params_keys is None:
                dirname = os.path.join(args.PATH, timestamp)
            else:
                dirname = os.path.join(args.PATH, args.params_keys, args.params_values)

            if args.training:
                gen.load_state_dict(torch.load(os.path.join(dirname, "gen0.pth")))
            else:
                assert args.saved_model_name is not None, "Please specify the saved model name"
                assert args.wandb == None, "No need to load anything to wand when only generating synthetic data"
                gen.load_state_dict(torch.load(os.path.join(args.saved_model_name, "gen0.pth")))
            
        gen.eval()

        with torch.no_grad():
            noise = torch.randn(args.num_generated, args.nz, 1, 1, device=device)
            normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
            to_pil = transforms.ToPILImage()

            fake = gen(noise).detach().cpu()
            fake = normalize(fake)

            dirname_npz_images = os.path.join(args.PATH_syn_data , 'npz_images', timestamp)
            dirname_npz_noise = os.path.join(args.PATH_syn_data , 'npz_noise', timestamp)
            dirname_png_images = os.path.join(args.PATH_syn_data , 'png_images', timestamp)

            os.makedirs(dirname_npz_images, exist_ok=True)
            np.savez(os.path.join(dirname_npz_images, "dcgan_synthetic_data.npz"), fake=fake)

            os.makedirs(dirname_npz_noise, exist_ok=True)
            np.savez(os.path.join(dirname_npz_noise, "dcgan_noise.npz"), noise=noise.cpu())

            os.makedirs(dirname_png_images, exist_ok=True)
            for i, img in enumerate(fake):
                pil_img = to_pil(img)
                save_path = os.path.join(dirname_png_images, f"image_{i}.png")
                pil_img.save(save_path)


def train_privGAN(genS, discS, private_disc, opt_gen, opt_disc, opt_private_disc, t, criterion, loss_fn, X_loaders, timestamp):

    #train generator and discriminator for gen_epochs
    batchCount = int(t // args.batch_size) + 1

    for epoch in range(args.num_epochs):

        d_t = np.zeros((args.N_splits, batchCount))
        dp_t = np.zeros(batchCount * args.N_splits)
        g_t = np.zeros(batchCount * args.N_splits)

        #for each dataloader in dataloader_N, train the discriminator
        for split in range(args.N_splits):

            l = list(range(args.N_splits))
            del(l[split])
            gen_y =  np.random.choice( l, t )
            gen_y = torch.tensor(gen_y, dtype=torch.float32).to(device)

            for i, (imgs, labels) in enumerate(X_loaders[split]):

                imgs = imgs.to(device)
                labels = labels.to(device)
                noise = torch.randn(imgs.shape[0], args.nz, 1, 1).to(device)
                fake = genS[split](noise)

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


                #train private discriminator. Label is the split index of the data. Goal is to determine which
                #generator produced the data. Hence, we use the split index as the label
                if epoch > args.dp_delay:
                    output = private_disc(fake).reshape(-1, args.N_splits)

                    loss_Dp = loss_fn(output, labels)

                    private_disc.zero_grad()
                    loss_Dp.backward(retain_graph=True)
                    opt_private_disc.step()

                    dp_t[i] = loss_Dp.item()


                # train generator. Goal is to fool the discriminator and the private discriminator.
                # Hence, we use different split index as the label for loss_fn, and ones as the label for
                #criterion.
                output1 = discS[split](fake).reshape(-1)
                output2 = private_disc(fake).reshape(-1, args.N_splits)
                labels_gen = gen_y[i*imgs.shape[0]:(i+1)*imgs.shape[0]] #labels is a different split index to fool the private discriminator

                lossG_1 = criterion(output1, torch.ones_like(output1))
                lossG_2 = args.privacy_ratio * loss_fn(output2, labels_gen.type(torch.LongTensor).to(device))
                lossG = lossG_1 + lossG_2

                genS[split].zero_grad()
                lossG.backward()
                opt_gen.step()

                g_t[i] = lossG.item()    

            #prepare labels for generator: random numbers as the goal is not only too fool 
            #the discriminator but also the private discriminator. A random lable makes the
            #generator generate images that are not too similar to the ones from the same split


        with torch.no_grad():
            print(f"Epoch {epoch} of {args.num_epochs} complete. Loss D: {d_t.mean()}, Loss DP: {dp_t.mean()}, Loss G: {g_t.mean()}")
            if args.wandb:
                wandb.log({"D_loss": d_t.mean(), "DP_loss": dp_t.mean(), "G_loss": g_t.mean()})


    #save the models
    if args.save_model:

        if args.params_keys is None:
            dirname = os.path.join(args.PATH, timestamp)
        else:
            dirname = os.path.join(args.PATH, args.params_keys, args.params_values)

        os.makedirs(dirname, exist_ok=True)
        for i in range(args.N_splits):
            torch.save(genS[i].state_dict(), os.path.join(dirname, f"gen{i}.pth") )
            torch.save(discS[i].state_dict(), os.path.join(dirname, f"disc{i}.pth") )
        torch.save(private_disc.state_dict(), os.path.join(dirname, "private_disc.pth"))


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  
  


if __name__ == '__main__':

    args.local_config= 'gan_models/dcgan/dcgan_config.yaml'

    if args.local_config is not None:

        with open(str(args.local_config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)

        if not args.ailab:
            import wandb #wandb is not supported on ailab server 
            
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")

    main()
