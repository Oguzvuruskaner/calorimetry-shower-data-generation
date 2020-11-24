import torch.nn as N
import torch


from src.helpers.AttrProxy import AttrProxy
from src.modules.LSTM.ParticleHead import ParticleHead


class ParticleGenerate(N.Module):


    def __init__(self,depth=4):
        super().__init__()

        self.depth = depth

        self.input = N.Sequential(
            N.Conv1d(3, 8, 5, 1, 2),
            N.LeakyReLU(),
            N.Conv1d(8, 16, 5, 1, 2),
            N.LeakyReLU(),
        )

        self.proxy = AttrProxy(self,"l_")

        for ind in range(self.depth):
            self.add_module(
                self.proxy(ind),
                N.Sequential(
                    N.Conv1d(16,16, 5, 1, 2),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            N.Conv1d(16,1, 5, 1, 2),
            ParticleHead()
        )

    def forward(self,state_down,particle,z_down):

        x = torch.cat([state_down,particle,z_down],dim=1)
        x = self.input(x)

        for ind in range(self.depth):
            module = self.proxy[ind]
            x = module(x)

        return self.out(x)