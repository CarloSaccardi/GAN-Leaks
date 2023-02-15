import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import wandb
import warnings
import yaml

from model import autoencoder, Generator, Discriminator, Dataset
from utils import generator_loss, autoencoder_loss, discriminator_loss, discriminator_accuracy, sample_transform, weights_init

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/PhisioNet/MIMIC/processed/out_binary.matrix'),
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

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")


parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--hidden_gen", type=int, default=128, help="dimensionality of the hidden layer of the generator")
parser.add_argument("--hidden_disc1", type=int, default=128, help="dimensionality of the first hidden layer of the discriminator")
parser.add_argument("--hidden_disc2", type=int, default=256, help="dimensionality of the second hidden layer of the discriminator")

parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=True, help="Minibatch averaging")

parser.add_argument("--training", type=bool, default=False, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--PATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/medGan'),
                    help="Training status")


opt = parser.parse_args()

device = torch.device("cuda" if opt.cuda else "cpu")

#### Train data loader ####
dataset_train_object = Dataset(data_file=opt.DATASETPATH, train=True, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

### Test data loader ####
dataset_test_object = Dataset(data_file=opt.DATASETPATH, train=False, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=False, num_workers=1, drop_last=True, sampler=samplerRandom)

# Initialize generator and discriminator
generator = Generator(z_dim=opt.latent_dim, hidden_size=opt.hidden_gen)
discriminator = Discriminator(input_size=dataset_train_object.data.shape[1], 
                                hidden_size1=opt.hidden_disc1, 
                                hidden_size2=opt.hidden_disc2,
                                minibatch_averaging=opt.minibatch_averaging)

# Initialize autoencoder and decoder
autoencoder_binary = autoencoder(dataset_train_object.data.shape[1], opt.latent_dim, binary=True)
autoencoder_count = autoencoder(dataset_train_object.data.shape[1], opt.latent_dim, binary=False)
decoder_binary = autoencoder_binary.decoder
decoder_count = autoencoder_count.decoder

# Put the models on given device
generator.to(device)
discriminator.to(device)
autoencoder_binary.to(device)
autoencoder_count.to(device)
decoder_binary.to(device)
decoder_count.to(device)

# Weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)
autoencoder_binary.apply(weights_init)
autoencoder_count.apply(weights_init)

# Optimizers
G_binary_params = [{'params': generator.parameters()},
            {'params': autoencoder_binary.parameters(), 'lr': 1e-4}]
G_count_params = [{'params': generator.parameters()},
            {'params': autoencoder_count.parameters(), 'lr': 1e-4}]      

optimizer_G_binary = optim.Adam(G_binary_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_G_count = optim.Adam(G_count_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

optimizer_AE_binary = optim.Adam(autoencoder_binary.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_AE_count = optim.Adam(autoencoder_count.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

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
            optimizer_A.zero_grad()

            a_loss.backward()
            optimizer_A.step()

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

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False

            # Generate a batch of images
            fake_samples = generatorModel(z)

            # uncomment if there is no autoencoder
            fake_samples = autoencoderDecoder(fake_samples)

            # Loss measures generator's ability to fool the discriminator
            g_loss = generator_loss(discriminatorModel(fake_samples).view(-1), valid)

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = True

            # reset gradients of discriminator
            optimizer_D.zero_grad()

            out_real = discriminatorModel(real_samples).view(-1)
            real_loss = discriminator_loss(out_real, valid)
            D_x = out_real.mean().item()
            real_loss.backward()

            # Measure discriminator's ability to classify real from generated samples
            # The detach() method constructs a new view on a tensor which is declared
            # not to need gradients, i.e., it is to be excluded from further tracking of
            # operations, and therefore the subgraph involving this view is not recorded.
            # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

            out_fake = discriminatorModel(fake_samples.detach()).view(-1)
            fake_loss = discriminator_loss(out_fake, fake)
            D_G_z = out_fake.mean().item()
            fake_loss.backward()

            # total loss and calculate the backprop
            d_loss = (real_loss + fake_loss) / 2

            # Optimizer step
            # d_loss.backward()
            optimizer_D.step()

        with torch.no_grad():

            # Variables
            real_samples_test = next(iter(dataloader_test))
            real_samples_test = Variable(real_samples_test.type(Tensor))
            # z = Variable(Tensor(np.random.normal(0, 1, (samples.shape[0], opt.latent_dim))))
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generator
            fake_samples_test = generatorModel(z)
            fake_samples_test = autoencoderDecoder(fake_samples_test)
            g_loss_test = generator_loss(discriminatorModel(fake_samples_test).view(-1), valid)

            # Discriminator
            out_real_test = discriminatorModel(real_samples_test).view(-1)
            real_loss_test = discriminator_loss(out_real_test, valid)
            # D_x_test = out_real_test.mean().item()
            accuracy_real_test = discriminator_accuracy(out_real_test, valid)

            out_fake_test = discriminatorModel(fake_samples_test.detach()).view(-1)
            fake_loss_test = discriminator_loss(out_fake_test, fake)
            # D_G_z_test = fake_loss_test.mean().item()
            accuracy_fake_test = discriminator_accuracy(out_fake_test, fake)

            # Accumulated loss
            d_loss_test = (real_loss_test + fake_loss_test) / 2

            # Test autoencoder
            reconst_samples_test = autoencoderModel(real_samples_test)
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

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
                'g_loss': g_loss,
                'd_loss': d_loss,
                'a_loss': a_loss,
            }, os.path.join(opt.PATH, "model_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            call("ls -d -1tr " + opt.PATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)



