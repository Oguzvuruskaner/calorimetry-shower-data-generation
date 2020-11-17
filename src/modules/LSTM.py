from pytorch_lightning import LightningModule

import torch
import torch.nn as N
import torch.optim as O

def init_with_zeros(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.constant_(0.0)
        m.bias.data.constant_(0)

def init_with_kaiming(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.kaiming_uniform_(0.0, 1.0)
        m.bias.data.normal_(0,0.005)

class LSTM(LightningModule):

    def __init__(self, latent_size=64, state_size=64,init_lr = 0.1, *args, **kwargs):


        super().__init__(*args, **kwargs)
        self.latent_size = latent_size
        self.state_size = state_size
        self.lr = init_lr

        self.particle_output = N.Sequential(
            N.Linear(latent_size+state_size+4,state_size),
            N.LayerNorm(state_size),
            N.Linear(state_size,4),
            N.Sigmoid()
        )

        self.particle_to_state = N.Sequential(
            N.Linear(4,state_size),
            N.LayerNorm(state_size),
            N.Linear(state_size,state_size),
            N.ReLU()
        )

        self.particle_forget = N.Sequential(
            N.Linear(4,state_size),
            N.LayerNorm(state_size),
            N.Linear(state_size,state_size),
            N.Sigmoid()
        )

        self.state_remember = N.Sequential(
            N.Linear(state_size,state_size),
            N.LayerNorm(state_size),
            N.Linear(state_size,state_size),
            N.Sigmoid()
        )

        self.criterion = N.MSELoss()

    def _init_layers(self):
        self.state_remember.apply(init_with_kaiming)
        self.particle_to_state.apply(init_with_kaiming)
        self.particle_output.apply(init_with_kaiming)
        self.particle_forget.apply(init_with_zeros)

    def forward(self,state,particle,z):

        state = self.state_remember(state) * state + self.particle_forget(particle) * self.particle_to_state(particle)

        return state,self.particle_output(torch.cat([state,particle,z])),z


    def training_step(self, batch, batch_idx):

        jet = batch.view(-1,4)

        state = torch.zeros(self.state_size).to(self.device)
        prev_particle = torch.zeros(4).to(self.device)
        z = torch.randn(self.latent_size).to(self.device)
        loss = 0
        for target in jet :

            state,tmp,z = self(state.detach(),prev_particle.detach(),z.detach())
            loss += self.criterion(tmp,target)
            prev_particle = tmp

        self.log("train_loss",loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = O.SGD(self.parameters(),lr=self.lr)
        return optimizer