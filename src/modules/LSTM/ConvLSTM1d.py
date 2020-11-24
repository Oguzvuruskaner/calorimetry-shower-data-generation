import torch.nn as N
import torch

from src.modules.LSTM.Down import Down
from src.modules.LSTM.ParticleGenerate import ParticleGenerate
from src.modules.LSTM.Up import Up
from src.modules.LSTM.NFilter import NFilter



class ConvLSTM1d(N.Module):

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.latent_size = kwargs.get("latent_size")
        self.state_size = kwargs.get("state_size")

        self.particle_filter = NFilter(3,1)
        self.particle_up = Up()

        self.z_down = Down()
        self.state_down = Down()


        self.particle_generate = ParticleGenerate()


    def generate(self,n = 45,device=torch.cuda.current_device()):

        self.eval()
        z = torch.randn((1,1,self.latent_size)).to(device)
        state = torch.zeros((1,1,self.latent_size)).to(device)
        particle = torch.zeros((1,1,4)).to(device)

        particles = torch.zeros((n,4))

        for i in range(n):

            state,particle,_ = self(state,particle,z)
            particles[i] = particle.squeeze().cpu()

        self.train()

        return particles

    def forward(self,state,particle,z):

        particle_up = self.particle_up(particle)

        state = state + particle_up * self.particle_filter(state,particle_up,z)

        state_down = self.state_down(state)
        z_down = self.z_down(z)

        particle = self.particle_generate(state_down,particle,z_down)

        return state,particle,z
