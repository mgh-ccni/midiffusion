import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from random import randint

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=9216):
        return input.reshape(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=16, z_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 256, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(256, 128, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(64, 256, kernel_size=4, stride=2),
            nn.Tanh(),
            Flatten()
        )

        self.fc0 = nn.Linear(9216, h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, 9216)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(9216, 256, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 16, kernel_size=6, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        h = self.fc0(h)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)

        z = self.decode(z)
        return z, mu, logvar


class AE(nn.Module):
    def __init__(self, image_channels=3, h_dim=16, z_dim=8):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc0 = nn.Linear(9216, h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, 9216)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(9216, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Tanh(),
        )

    def bottleneck(self, h):
        h = self.fc0(h)
        mu = self.fc1(h)
        z = mu
        return z

    def encode(self, x):
        h = self.encoder(x)
        z= self.bottleneck(h)
        return z

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z
