import torch.nn as N

from src.modules.GAN.Sigma import Sigma
from src.modules.GAN.Up import Up


class StateReducer(N.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.up = Up(*args,**kwargs)
        self.sigma = Sigma(*args,**kwargs)


    def forward(self,state,particle):

        particle_up = self.up(particle)
        return state + particle_up*self.sigma(particle)