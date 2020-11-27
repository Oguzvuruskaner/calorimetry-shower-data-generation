import torch.nn as N
import torch

from src.modules.GAN.BasicNetwork import BasicNetwork
from src.modules.GAN.MinibatchDiscrimination import MinibatchDiscrimination
from src.modules.GAN.StateNetwork import StateNetwork
from src.modules.LSTM.GlobalAveragePooling1d import GlobalAveragePooling1d


class Discriminator(BasicNetwork):



    def __init__(self, *args, **kwargs):


        super().__init__(*args,**kwargs)

        self.repeat = kwargs.get("repeat",16)

        self.depth = kwargs.get("depth",4)
        self.state_net = StateNetwork(*args,**kwargs,out_filter=32)

        self.input = N.Sequential(
            N.Conv1d(self.repeat+32,64,3,1,1),
            N.LeakyReLU(),
        )

        for i in range(self.depth):
            self.add_module(
                self.proxy(i),
                N.Sequential(
                    N.Conv1d(64,64,3,1,1),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            GlobalAveragePooling1d(),
            MinibatchDiscrimination(64,64,64),
            N.Linear(128,1),
            N.Sigmoid()
        )


    def forward(self,state,p):

        state = self.state_net(state)
        p = p.repeat(1,self.repeat,1)
        x = torch.cat([state,p],dim=1)
        return super().forward(x)
