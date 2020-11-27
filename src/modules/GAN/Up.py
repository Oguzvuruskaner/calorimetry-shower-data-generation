import torch.nn as N
import math

from src.modules.GAN.BasicNetwork import BasicNetwork

class Up(BasicNetwork):

    def __init__(self,*args,**kwargs):


        super().__init__(*args,**kwargs)

        self.root_filter = kwargs.get("root_filter",16)
        self.latent_size = kwargs.get("latent_size",128)
        self.depth = int(math.log2(self.latent_size // 4))


        self.input = N.Sequential(
            N.Conv1d(1, self.root_filter//2, 3, 1, 1),
            N.LeakyReLU(),
            N.Conv1d(self.root_filter//2, self.root_filter, 3, 1, 1),
            N.LeakyReLU(),
            N.Conv1d(self.root_filter,self.get_width(self.root_filter*2**self.depth),3,1,1),
            N.LeakyReLU()
        )

        for i in range(self.depth):
            self.add_module(
                self.proxy(self.depth-i-1),
                N.Sequential(
                    N.ConvTranspose1d(self.get_width(self.root_filter*2**(i+1)),self.get_width(self.root_filter*2**i),4,2,1),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            N.Conv1d(self.root_filter,1,5,1,2),
            N.LeakyReLU()
        )



