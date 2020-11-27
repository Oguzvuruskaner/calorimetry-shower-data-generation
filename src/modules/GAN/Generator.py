import torch.nn as N
import torch

import math

from src.modules.GAN.BasicNetwork import BasicNetwork
from src.modules.GAN.MinibatchDiscrimination import MinibatchDiscrimination
from src.modules.GAN.StateNetwork import StateNetwork
from src.modules.LSTM.GlobalAveragePooling1d import GlobalAveragePooling1d
from src.modules.helpers.ParticleHead import ParticleHead


class Generator(BasicNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.latent_size = kwargs.get("latent_size",128)
        self.depth = int(math.log2(self.latent_size // 4))
        self.state_net = StateNetwork(*args, **kwargs, out_filter=32)

        self.input = N.Sequential(
            N.Conv1d(2, self.root_filter//2, 3, 1, 1),
            N.LeakyReLU(),
            N.Conv1d(self.root_filter // 2, self.root_filter, 3, 1, 1),
            N.LeakyReLU(),
        )

        for i in range(self.depth):
            self.add_module(
                self.proxy(i),
                N.Sequential(
                    N.Conv1d(self.get_width(self.root_filter*2**i),self.get_width(self.root_filter*2**(i+1)),3,2,1),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            N.Conv1d( self.get_width(self.root_filter * 2 ** self.depth),1, 3, 1,1),
            ParticleHead()
        )

    def forward(self, x, z):

        x = torch.cat([x, z], dim=1)
        return super().forward(x)
