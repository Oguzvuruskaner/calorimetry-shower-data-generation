from pytorch_lightning import LightningModule

import torch
import torch.nn as N
import torch.optim as O

from adabound import AdaBound

from src.modules.LSTM.ConvLSTM1d import ConvLSTM1d


class LSTMLightning(LightningModule):

    def __init__(self,  *args, **kwargs):

        super().__init__()

        self.latent_size = kwargs.get("latent_size",64)
        self.state_size = kwargs.get("state_size",64)
        self.lr = kwargs.get("lr",10e-3)

        self.model = ConvLSTM1d(latent_size=self.latent_size,state_size=self.state_size)

        self.criterion = N.MSELoss()


    def training_step(self, batch, batch_idx):

        jet = batch.view(-1,len(batch),1,4)

        state = torch.zeros((len(batch),1,self.state_size),requires_grad=False).to(self.device)
        prev_particle = torch.zeros((len(batch),1,4)).to(self.device)
        z = torch.randn((len(batch),1,self.latent_size),requires_grad=False).to(self.device)
        loss = 0

        for target in jet :

            state,tmp,z = self.model(state,prev_particle.detach(),z)
            loss += self.criterion(tmp,target)
            prev_particle = tmp


        self.log("train_loss",loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = AdaBound(self.parameters(),lr=self.lr)
        lr_scheduler = O.lr_scheduler.ReduceLROnPlateau(optimizer,threshold=10e-8,factor=.1,cooldown=5,patience=10,verbose=True)

        return {
            "optimizer":optimizer,
            "monitor":"train_loss",
            "lr_scheduler":lr_scheduler
        }
