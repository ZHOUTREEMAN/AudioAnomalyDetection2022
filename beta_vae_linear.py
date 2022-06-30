from typing import *

import torch
from base import BaseVAE
from torch import nn, Tensor
from torch.nn import functional as F


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        output_size = in_channels

        # Build Encoder
        self.input_to_hidden1 = nn.Linear(in_channels, hidden_dims[0])
        self.hidden1_to_hidden2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.hidden2_to_hidden3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.hidden3_to_hidden4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.fc_mu = nn.Linear(hidden_dims[3], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[3], latent_dim)
        nn.init.xavier_uniform_(self.fc_mu.weight)  # 为了通过网络层时，输入和输出的方差相同 服从均匀分布
        nn.init.xavier_uniform_(self.fc_var.weight)  # 为了通过网络层时，输入和输出的方差相同

        # Build Decoder
        self.latent_to_hidden4 = nn.Linear(latent_dim, hidden_dims[3])
        self.hidden4_to_hidden3 = nn.Linear(hidden_dims[3], hidden_dims[2])
        self.hidden3_to_hidden2 = nn.Linear(hidden_dims[2], hidden_dims[1])
        self.hidden2_to_hidden1 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.hidden1_to_output = nn.Linear(hidden_dims[0], output_size)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        hidden1 = self.ReLU(self.input_to_hidden1(input))
        hidden2 = self.ReLU(self.hidden1_to_hidden2(hidden1))
        hidden3 = self.ReLU(self.hidden2_to_hidden3(hidden2))
        hidden4 = self.ReLU(self.hidden3_to_hidden4(hidden3))
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(hidden4)
        log_var = self.fc_var(hidden4)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        hidden4 = self.ReLU(self.latent_to_hidden4(z))
        hidden3 = self.ReLU(self.hidden4_to_hidden3(hidden4))
        hidden2 = self.ReLU(self.hidden3_to_hidden2(hidden3))
        hidden1 = self.ReLU(self.hidden2_to_hidden1(hidden2))
        result = self.hidden1_to_output(hidden1)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
