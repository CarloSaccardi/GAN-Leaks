import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from subprocess import call
import time
import matplotlib.pyplot as plt
import datetime
import sys

import wandb
import warnings
import yaml

from model import  CustomDataset, Autoencoder, Generator, Discriminator
from utils import generator_loss, autoencoder_loss, discriminator_loss, discriminator_accuracy, sample_transform, init_weights

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('data\mini_MIMIC_III\mini_MIMIC_III.csv'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--n_epochs_pretrain", type=int, default=100,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=2000, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--cuda", type=bool, default=False, help="CUDA activation")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--hidden_gen", type=int, default=128, help="dimensionality of the hidden layer of the generator")
parser.add_argument("--hidden_disc1", type=int, default=128, help="dimensionality of the first hidden layer of the discriminator")
parser.add_argument("--hidden_disc2", type=int, default=256, help="dimensionality of the second hidden layer of the discriminator")
parser.add_argument("--binary", type=bool, default=True, help="Binary or count data")
parser.add_argument("--opt.generate_N", type=int, default=100, help="Number of samples to generate")

parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=True, help="Minibatch averaging")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--PATH", type=str, default=os.path.expanduser('gan_models/medgan/model_save'),
                    help="Training status")
parser.add_argument('--local_config', default=None, help='path to config file')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")
parser.add_argument("--model_directory", type=str, default=None, help="Directory to saved model")


opt = parser.parse_args()

#### Train and test data loader ####
train_dataset = CustomDataset(opt.DATASETPATH, train=True)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=train_dataset, replacement=True)
dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=samplerRandom)

test_dataset = CustomDataset(opt.DATASETPATH , train=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=test_dataset, replacement=True)
dataloader_test = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler=samplerRandom)

def main():

    #sys.dont_write_bytecode = True # To avoid creating .pyc files

    now = datetime.datetime.now() # To create a unique folder for each run
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # To create a unique folder for each run

    device = torch.device("cuda" if opt.cuda else "cpu") # Set device

    # Initialize generator, discriminator, autoencoder and decoder
    
    generator = Generator(z_dim = opt.latent_dim, 
                          hidden_size = opt.hidden_gen)
    
    discriminator = Discriminator(input_size=train_dataset.data.shape[1], 
                                  hidden_size1=opt.hidden_disc1, 
                                  hidden_size2=opt.hidden_disc2, 
                                  minibatch_average=opt.minibatch_averaging)
    
    autoencoder = Autoencoder(input_size=train_dataset.data.shape[1], 
                              hidden_size=opt.hidden_gen, 
                              binary=opt.binary)
    

    # Put the models on given device
    generator.to(device)
    discriminator.to(device)
    autoencoder.to(device)
    
    decoder = autoencoder.decode

    # Weight initialization
    init_weights(autoencoder, method='xavier_uniform')
    init_weights(generator, method='normal')
    init_weights(discriminator, method='normal')

    # Optimizers
    G_params = [{'params': generator.parameters()},
                {'params': autoencoder.parameters(), 'lr': 1e-4}]
    optimizer_G = optim.Adam(G_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    optimizer_AE= optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)


    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    if opt.training:

        assert opt.generate == False, "You can not generate data while training"

        for epoch_pre in range(opt.n_epochs_pretrain):
            for i, samples in enumerate(dataloader_train):

                # Configure input
                real_samples = Variable(samples.type(Tensor))

                # Generate a batch of images
                recons_samples = autoencoder(real_samples)

                # Loss measures generator's ability to fool the discriminator
                a_loss = autoencoder_loss(recons_samples, real_samples, binary=opt.binary)

                # # Reset gradients (if you uncomment it, it would be a mess. Why?!!!!!!!!!!!!!!!)
                optimizer_AE.zero_grad()

                a_loss.backward()
                optimizer_AE.step()

                batches_done = epoch_pre * len(dataloader_train) + i
                if batches_done % opt.sample_interval == 0:
                    print(
                        "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                        % (epoch_pre + 1, opt.n_epochs_pretrain, i, len(dataloader_train), a_loss.item())
                        , flush=True)

        for epoch in range(opt.n_epochs):
            epoch_start = time.time()
            for i, samples in enumerate(dataloader_train):

                # Configure input
                real_samples = Variable(samples.type(Tensor))

                # Sample noise as generator input
                z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # reset gradients of discriminator
                optimizer_D.zero_grad()

                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = True

                # Measure discriminator's ability to classify real from generated samples
                # The detach() method constructs a new view on a tensor which is declared
                # not to need gradients, i.e., it is to be excluded from further tracking of
                # operations, and therefore the subgraph involving this view is not recorded.
                # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                out_real = discriminator(real_samples).view(-1)
                out_fake = discriminator(decoder(generator(z)).detach()).view(-1)
                d_loss = discriminator_loss(out_real, out_fake)
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------

                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = False

                # Generate a batch of images
                fake_samples = generator(z)

                # uncomment if there is no autoencoder
                fake_samples = decoder(fake_samples)

                # Loss measures generator's ability to fool the discriminator
                g_loss = generator_loss(discriminator(fake_samples).view(-1))
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

            with torch.no_grad():

                # Variables
                real_samples_test = next(iter(dataloader_test))
                real_samples_test = Variable(real_samples_test.type(Tensor))
                z = torch.randn(real_samples_test.shape[0], opt.latent_dim, device=device)

                # Generator
                fake_samples_test = generator(z)
                fake_samples_test = decoder(fake_samples_test)
                g_loss_test = generator_loss(discriminator(fake_samples_test).view(-1))

                # Discriminator
                out_real_test = discriminator(real_samples_test).view(-1)
                out_fake_test = discriminator(fake_samples_test.detach()).view(-1)
                d_loss_test = discriminator_loss(out_real_test, out_fake_test)

                accuracy_real_test = discriminator_accuracy(predicted=out_real_test, y_true=True)
                accuracy_fake_test = discriminator_accuracy(predicted=out_fake_test, y_true=False)

                # Test autoencoder
                reconst_samples_test = autoencoder(real_samples_test)
                a_loss_test = autoencoder_loss(reconst_samples_test, real_samples_test, binary=opt.binary)

                # every 10 epochs print loss and accuracy_real and accuracy_fake
                if (epoch + 1) % 10 == 0:
                    print(
                        "[Epoch %d/%d] [D loss: %.3f] [G loss: %.3f] [A loss: %.3f] [Real acc: %.3f] [Fake acc: %.3f] "
                        % (epoch + 1, opt.n_epochs, d_loss.item(), g_loss.item(), a_loss.item(), 
                            accuracy_real_test, accuracy_fake_test)
                        , flush=True)


                # push metrics to wandb every epoch 
                wandb.log({"epoch": epoch + 1, "d_loss": d_loss.item(), "g_loss": g_loss.item(), "a_loss": a_loss.item(),
                            "accuracy_real": accuracy_real_test, "accuracy_fake": accuracy_fake_test })
                
                # save model to wandb
                if epoch  ==  opt.n_epochs - 1:
                    dirname = os.path.join(opt.PATH, timestamp)
                    os.makedirs(dirname, exist_ok=True)
                    torch.save(generator.state_dict(), os.path.join(dirname, "generator.pth"))
                    torch.save(autoencoder.decoder.state_dict(), os.path.join(dirname, "decoder.pth"))
                    torch.save(autoencoder.state_dict(), os.path.join(dirname, "autoencoder.pth"))
                    torch.save(discriminator.state_dict(), os.path.join(dirname, "discriminator.pth"))
                    
    if opt.generate:

        assert opt.training == False, "You can't generate data if you are training the model"

        # Check cuda
        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device BUT it is not in use...")

        # Activate CUDA
        device = torch.device("cuda:0" if opt.cuda else "cpu")

        #####################################
        #### Load model and optimizer #######
        #####################################

        # random seed
        np.random.seed(1234)

        # Initialize generator and discriminator
        generator = Generator(z_dim = opt.latent_dim, 
                            hidden_size = opt.hidden_gen)
        
        discriminator = Discriminator(input_size=train_dataset.data.shape[1], 
                                    hidden_size1=opt.hidden_disc1, 
                                    hidden_size2=opt.hidden_disc2, 
                                    minibatch_average=opt.minibatch_averaging)
        
        autoencoder = Autoencoder(input_size=train_dataset.data.shape[1], 
                                hidden_size=opt.hidden_gen, 
                                binary=opt.binary)

        # Loading the checkpoint
        assert os.path.exists(opt.model_directory), "The model directory does not exist"
        model_dir = opt.model_directory
        checkpoint_generator = torch.load(os.path.join(model_dir, "generator.pth"))
        checkpoint_autoencoder = torch.load(os.path.join(model_dir, "autoencoder.pth"))
        checkpoint_autoencoder_decoder = torch.load(os.path.join(model_dir, "decoder.pth"))
        checkpoint_discriminator = torch.load(os.path.join(model_dir, "discriminator.pth"))

        # Load models
        generator.load_state_dict(checkpoint_generator)
        autoencoder.load_state_dict(checkpoint_autoencoder)
        autoencoder.decoder.load_state_dict(checkpoint_autoencoder_decoder)
        discriminator.load_state_dict(checkpoint_discriminator)

        # insert weights [required]
        generator.to(device)
        autoencoder.to(device)
        discriminator.to(device)
        decoder = autoencoder.decoder

        #######################################################
        #### Load real data and generate synthetic data #######
        #######################################################

        # Load real data
        real_samples = dataloader_train.dataset.data

        # Generate a batch of samples
        z = torch.randn(opt.generate_N, opt.latent_dim, device=device)
        gen_samples_tensor = generator(z)
        gen_samples_decoded = decoder(gen_samples_tensor)
        gen_samples = gen_samples_decoded.detach().cpu().numpy()

        gen_samples[gen_samples >= 0.5] = 1.0
        gen_samples[gen_samples < 0.5] = 0.0

        # Trasnform Object array to float
        gen_samples = gen_samples.astype(np.float32)

        # ave synthetic data
        np.save(os.path.join(opt.PATH, "synthetic.npy"), gen_samples, allow_pickle=False)

    if opt.evaluate:
        # Load synthetic data
        gen_samples = np.load(os.path.join(opt.PATH, "synthetic.npy"), allow_pickle=False)

        # Load real data
        real_samples = train_dataset.return_data()[0:gen_samples.shape[0], :]

        # Dimenstion wise probability
        prob_real = np.mean(real_samples, axis=0)
        prob_syn = np.mean(gen_samples, axis=0)

        p1 = plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label="medGAN")
        x_max = max(np.max(prob_real), np.max(prob_syn))
        x = np.linspace(0, x_max + 0.1, 1000)
        p2 = plt.plot(x, x, linestyle='-', color='k', label="Ideal")  # solid
        plt.tick_params(labelsize=12)
        plt.legend(loc=2, prop={'size': 15})
        # plt.title('Scatter plot p')
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.show()


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  


if __name__ == '__main__':

    if opt.local_config is not None:
        with open(str(opt.local_config), "r") as f:
            config = yaml.safe_load(f)
        update_args(opt, config)
        if opt.wandb:
            wandb_config = vars(opt)
            run = wandb.init(project=str(opt.wandb), entity="thesis_carlo", config=wandb_config)
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")

    main()
    a=1


