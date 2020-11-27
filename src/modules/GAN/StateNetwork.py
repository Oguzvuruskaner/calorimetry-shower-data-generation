import torch.nn as N

from src.modules.GAN.BasicNetwork import BasicNetwork
import math



class StateNetwork(BasicNetwork):



    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.out_filter = kwargs.get("out_filter",32)
        self.latent_size = kwargs.get("latent_size",128)
        self.depth = int(math.log2(self.latent_size // 4))

        self.input = N.Sequential(
            N.Conv1d(1, self.root_filter // 2, 5, 1, 2),
            N.LeakyReLU(),
            N.Conv1d(self.root_filter // 2, self.root_filter, 5, 1, 2),
            N.LeakyReLU()
        )

        for i in range(self.depth):
            self.add_module(
                self.proxy(i),
                N.Sequential(
                    N.Conv1d(self.get_width(self.root_filter*2**(i)),self.get_width(self.root_filter*2**(i+1)),3,2,1),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            N.Conv1d(self.get_width(self.root_filter * 2 ** self.depth ),self.out_filter,3, 1,1),
            N.LeakyReLU()
        )

