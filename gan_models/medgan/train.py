import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from subprocess import call
import time

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
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--cuda", type=bool, default=False,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=False,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=0, help="Number of GPUs in case of multiple GPU")


parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--hidden_gen", type=int, default=128, help="dimensionality of the hidden layer of the generator")
parser.add_argument("--hidden_disc1", type=int, default=128, help="dimensionality of the first hidden layer of the discriminator")
parser.add_argument("--hidden_disc2", type=int, default=256, help="dimensionality of the second hidden layer of the discriminator")
parser.add_argument("--binary", type=bool, default=True, help="Binary or count data")

parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=True, help="Minibatch averaging")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--PATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/medGan'),
                    help="Training status")
parser.add_argument('--local_config', default=None, help='path to config file')
parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")


opt = parser.parse_args()

device = torch.device("cuda" if opt.cuda else "cpu")

#### Train and test data loader ####
train_dataset = CustomDataset(opt.DATASETPATH, train=True)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=train_dataset, replacement=True)
dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=samplerRandom)

test_dataset = CustomDataset(opt.DATASETPATH , train=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=test_dataset, replacement=True)
dataloader_test = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler=samplerRandom)

def main():

    # Initialize generator, discriminator, autoencoder and decoder
    
    generator = Generator(z_dim = opt.latent_dim, 
                          hidden_size = opt.hidden_gen)
    
    discriminator = Discriminator(input_size=train_dataset.data.shape[1], 
                                  hidden_size1=opt.hidden_disc1, 
                                  hidden_size2=opt.hidden_disc2, 
                                  minibatch_avarage=opt.minibatch_averaging)
    
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
        for epoch_pre in range(opt.n_epochs_pretrain):
            for i, samples in enumerate(dataloader_train):

                # Configure input
                real_samples = Variable(samples.type(Tensor))

                # Generate a batch of images
                recons_samples = autoencoder(real_samples)

                # Loss measures generator's ability to fool the discriminator
                a_loss = autoencoder_loss(recons_samples, real_samples)

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

                # Adversarial ground truths
                valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

                # Configure input
                real_samples = Variable(samples.type(Tensor))

                # Sample noise as generator input
                z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                # -----------------
                #  Train Generator
                # -----------------

                # We’re supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step().
                #
                # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
                # accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So,
                # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
                #
                # Because of this, when you start your training loop, ideally you should zero out the gradients so
                # that you do the parameter update correctly. Else the gradient would point in some other direction
                # than the intended direction towards the minimum (or maximum, in case of maximization objectives).

                # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between
                # minibatches, you have to zero them out at the start of a new minibatch. This is exactly like how a general
                # (additive) accumulator variable is initialized to 0 in code.

                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = False

                # Generate a batch of images
                fake_samples = generator(z)

                # uncomment if there is no autoencoder
                fake_samples = decoder(fake_samples)

                # Loss measures generator's ability to fool the discriminator
                g_loss = generator_loss(discriminator(fake_samples).view(-1), valid)

                # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                for p in discriminator.parameters():  # reset requires_grad
                    p.requires_grad = True

                # reset gradients of discriminator
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                # The detach() method constructs a new view on a tensor which is declared
                # not to need gradients, i.e., it is to be excluded from further tracking of
                # operations, and therefore the subgraph involving this view is not recorded.
                # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                out_real = discriminator(real_samples).view(-1)
                out_fake = discriminator(fake_samples.detach()).view(-1)
                d_loss = discriminator_loss(out_real, out_fake)
                D_x = out_real.mean().item()
                D_G_z = out_fake.mean().item()
                d_loss.backward()

                optimizer_D.step()

            with torch.no_grad():

                # Variables
                real_samples_test = next(iter(dataloader_test))
                real_samples_test = Variable(real_samples_test.type(Tensor))
                z = torch.randn(real_samples_test.shape[0], opt.latent_dim, device=device)

                valid_test = Variable(Tensor(real_samples_test.shape[0]).fill_(1.0), requires_grad=False)
                fake_test = Variable(Tensor(real_samples_test.shape[0]).fill_(0.0), requires_grad=False)

                # Generator
                fake_samples_test = generator(z)
                fake_samples_test = decoder(fake_samples_test)
                g_loss_test = generator_loss(discriminator(fake_samples_test).view(-1), valid_test)

                # Discriminator
                out_real_test = discriminator(real_samples_test).view(-1)
                real_loss_test = discriminator_loss(out_real_test, valid_test)
                # D_x_test = out_real_test.mean().item()
                accuracy_real_test = discriminator_accuracy(out_real_test, valid_test)

                out_fake_test = discriminator(fake_samples_test.detach()).view(-1)
                fake_loss_test = discriminator_loss(out_fake_test, fake_test)
                #loss = -torch.mean(torch.log(outputs_real + 1e-12)) - torch.mean(torch.log(1. - outputs_fake + 1e-12))
                # D_G_z_test = fake_loss_test.mean().item()
                accuracy_fake_test = discriminator_accuracy(out_fake_test, fake_test)

                # Accumulated loss
                d_loss_test = (real_loss_test + fake_loss_test) / 2

                # Test autoencoder
                reconst_samples_test = autoencoder(real_samples_test)
                a_loss_test = autoencoder_loss(reconst_samples_test, real_samples_test)

            print(
                "TRAIN: [Epoch %d/%d] [Batch %d/%d] [D loss: %.2f] [G loss: %.2f] [A loss: %.2f]"
                % (epoch + 1, opt.n_epochs, i, len(dataloader_train), d_loss.item(), g_loss.item(), a_loss.item())
                , flush=True)

            print(
                "TEST: [Epoch %d/%d] [Batch %d/%d] [D loss: %.2f] [G loss: %.2f] [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]"
                % (epoch + 1, opt.n_epochs, i, len(dataloader_train), d_loss_test.item(), g_loss_test.item(),
                a_loss_test.item(), accuracy_real_test,
                accuracy_fake_test)
                , flush=True)

            # End of epoch
            epoch_end = time.time()
            if opt.epoch_time_show:
                print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)




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


