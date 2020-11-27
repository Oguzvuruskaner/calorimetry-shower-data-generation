from collections import OrderedDict

from pytorch_lightning import LightningModule

import torch
import torch.nn as N
import torch.optim as O

from adabound import AdaBound

from src.modules.GAN.Discriminator import Discriminator
from src.modules.GAN.Generator import Generator
from src.modules.GAN.StateReducer import StateReducer


class GANLightning(LightningModule):

    def __init__(self,  *args, **kwargs):

        super().__init__()

        self.disc_lr = kwargs.get("disc_lr",10e-3)
        self.gen_lr = kwargs.get("gen_lr",10e-4)
        self.latent_size = kwargs.get("latent_size",128)
        self.max_particle = kwargs.get("max_particle",1500)

        self.generator     = Generator(*args,**kwargs)
        self.discriminator = Discriminator(*args,**kwargs)
        self.state         = StateReducer(*args,**kwargs)

        self.real_label = torch.ones((1,1)).to(self.device)
        self.fake_label = torch.zeros((1,1)).to(self.device)

        self.criterion = N.BCELoss()



    def training_step(self, batch, batch_idx,optimizer_idx):


        batch = batch.permute([1,0,2])

        # disc_optimizer
        if optimizer_idx == 0:

            state = torch.zeros((1,1,self.latent_size)).to(self.device)

            real_loss = 0
            fake_loss = 0

            for particle in batch:

                particle = particle.view(1,1,-1)

                result = self.discriminator(state,particle)
                real_loss += self.criterion(result,self.real_label.to(self.device))
                state = self.state(state,particle)

            state = torch.zeros((1,1,self.latent_size)).to(self.device)

            for i in range(self.max_particle):

                latent = torch.randn((1,1,self.latent_size)).to(self.device)

                particle = self.generator(state,latent)
                result = self.discriminator(state,particle.detach())
                fake_loss += self.criterion(result,self.fake_label.to(self.device))
                state = self.state(state,particle)

            total_loss = fake_loss + real_loss

            self.log("disc_train_loss",total_loss.item())
            self.log("disc_real_loss",real_loss.item(),prog_bar=True)
            self.log("disc_fake_loss",fake_loss.item(),prog_bar=True)


            return total_loss


        elif optimizer_idx == 1:

            gen_loss = 0
            state = torch.zeros((1,1,self.latent_size)).to(self.device)

            for i in range(self.max_particle):
                latent = torch.randn((1, 1, self.latent_size)).to(self.device)

                particle = self.generator(state, latent)
                result = self.discriminator(state, particle)
                gen_loss  += self.criterion(result, self.real_label.to(self.device))
                state = self.state(state, particle)


            self.log("gen_loss", gen_loss.item(),prog_bar=True)


            return gen_loss


    def configure_optimizers(self):
        disc_optimizer = AdaBound([
            {
                "params":self.discriminator.parameters(),
            },
            {
                "params":self.state.parameters()
            }
        ],lr=self.disc_lr)
        disc_lr_scheduler = O.lr_scheduler.ReduceLROnPlateau(disc_optimizer,threshold=10e-8,factor=.1,cooldown=5,patience=10,verbose=True)

        gen_optimizer = AdaBound([
                {
                    "params":self.generator.parameters(),
                },
                {
                    "params": self.state.parameters()
                }
        ], lr=self.gen_lr)
        gen_lr_scheduler = O.lr_scheduler.ReduceLROnPlateau(gen_optimizer, threshold=10e-8, factor=.1, cooldown=5,
                                                        patience=10, verbose=True)


        return [disc_optimizer,gen_optimizer],[
            {
                "scheduler":disc_lr_scheduler,
                "monitor":"disc_train_loss",
                "interval":"epoch"
            },
            {
                "scheduler": gen_lr_scheduler,
                "monitor": "gen_train_loss",
                "interval": "epoch"
            },

        ]


    def generate(self,particle_limit = None):


        limit = particle_limit or self.max_particle


        particles = torch.zeros((limit,4))

        state = torch.zeros((1, 1, self.latent_size)).to(self.device)

        for i in range(limit):
            latent = torch.randn((1, 1, self.latent_size)).to(self.device)

            particle = self.generator(state, latent)
            state = self.state(state, particle)

            particles[i] = particle.squeeze()

        return particles