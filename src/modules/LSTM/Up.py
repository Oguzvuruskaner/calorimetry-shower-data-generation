import torch.nn as N

from src.helpers.AttrProxy import AttrProxy
from src.modules.LSTM.GlobalAveragePooling1d import GlobalAveragePooling1d


class Up(N.Module):

    def __init__(self,dropout_rate = 0.5, depth=4):
        super().__init__()

        self.depth = depth
        self.dropout_rate = dropout_rate

        self.proxy = AttrProxy(self,"l_")

        self.input = N.Sequential(
            N.Conv1d(1, 4, 5, 1, 2),
            N.LeakyReLU(),
            N.Conv1d(4, 16, 5, 1, 2),
            N.LeakyReLU()
        )

        for i in range(self.depth):
            self.add_module(
                self.proxy(i),
                N.Sequential(
                    N.ConvTranspose1d(16,16,4,2,1),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            N.Conv1d(16,1,5,1,2),
            N.LeakyReLU()
        )



    def forward(self,x):


        x = self.input(x)

        for ind in range(self.depth):

            module = self.proxy[ind]
            x = module(x)

        return self.out(x)