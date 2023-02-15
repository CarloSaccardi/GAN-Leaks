#pytorch implementation of medgan
#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys


class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, binary=False):
        super(autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        ## encoder
        self.encoder_layers = [nn.Linear(self.input_size, self.hidden_size)]
        if binary:
            self.encoder_layers.append(nn.Tanh())
        else:
            self.encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*self.encoder_layers)

        ## decoder
        self.decoder_layers = [nn.Linear(self.hidden_size, self.input_size)]
        if binary:
            self.decoder_layers.append(nn.Sigmoid())
        else:
            self.decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*self.decoder_layers)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_size):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.genDim = 128
        self.gen_block1 = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, eps=0.001, momemntum=0.01),
            nn.ReLU()
        )
        self.gen_block2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.genDim),
            nn.BatchNorm1d(self.genDim, eps=0.001, momemntum=0.01),
            nn.Tanh()
        )

    def forward(self, x):

        #layer 1
        z = x
        temp = self.gen_block1(x)
        out1 = z + temp

        #layer 2
        z = out1
        temp = self.gen_block2(out1)
        out2 = z + temp

        return out2


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, minibatch_avarage=False):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1 # 264
        self.hidden_size2 = hidden_size2 # 128
        self.minibatch_avarage = minibatch_avarage

        ma_coeff = 1 if minibatch_avarage else 2

        self.disc = nn.Sequential(
            nn.Linear(ma_coeff * self.input_size, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        if self.minibatch_avarage:
            ### minibatch averaging
            x_mean = torch.mean(x, dim=0).repeat(x.size(0), 1)
            x = torch.cat([x, x_mean], dim=1)
        
        return self.disc(x)


class Dataset:
    def __init__(self, data_file, train=None, transform=None):

        # Transform
        self.transform = transform
        self.train = train

        # load data here
        self.input = np.load(os.path.expanduser(data_file), allow_pickle=True)
        self.sampleSize = self.input.shape[0]
        self.featureSize = self.input.shape[1]

        # Split train-test
        indices = np.random.permutation(self.sampleSize)
        training_idx, test_idx = indices[:int(0.9 * self.sampleSize)], indices[int(0.9 * self.sampleSize):]
        if self.train == True:
            self.data = self.input[training_idx, :]
        else:
            self.data = self.input[test_idx, :]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
           pass

        return torch.from_numpy(sample)

