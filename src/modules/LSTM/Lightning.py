from pytorch_lightning import LightningModule

import torch
import torch.nn as N
import torch.optim as O

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

        jet = batch.view(-1,4)

        state = torch.zeros((1,1,self.state_size)).to(self.device)
        prev_particle = torch.zeros((1,1,4)).to(self.device)
        z = torch.randn((1,1,self.latent_size)).to(self.device)
        loss = 0

        for ind,target in enumerate(jet) :

            if ind < 40:

                state,tmp,z = self.model(state.detach(),prev_particle.detach(),z.detach())
                loss += self.criterion(tmp,target)
                prev_particle = tmp

            else:
                break

        self.log("train_loss",loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = O.Adam(self.parameters(),lr=self.lr)
        lr_scheduler = O.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=3,verbose=True)

        return {
            "optimizer":optimizer,
            "monitor":"train_loss",
            "lr_scheduler":lr_scheduler
        }
