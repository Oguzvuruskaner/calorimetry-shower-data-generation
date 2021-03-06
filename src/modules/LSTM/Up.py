import torch.nn as N

from src.helpers.AttrProxy import AttrProxy
from src.modules.helpers.DropoutModule import DropoutModule
from src.modules.helpers.MaxWidth import MaxWidth


class Up(N.Module,DropoutModule,MaxWidth):

    def __init__(self,*args,**kwargs):

        dropout_rate = kwargs.get("dropout_rate",.5)
        max_width = kwargs.get("max_width",128)

        super().__init__()
        DropoutModule.__init__(self,dropout_rate=dropout_rate)
        MaxWidth.__init__(self,max_width=max_width)

        self.max_width = kwargs.get("max_width",128)
        self.depth = kwargs.get("depth",4)
        self.root_filter = kwargs.get("root_filter",16)


        self.proxy = AttrProxy(self,"l_")

        self.input = N.Sequential(
            N.Conv1d(1, self.root_filter//2, 5, 1, 2),
            N.LeakyReLU(),
            N.Conv1d(self.root_filter//2, self.root_filter, 5, 1, 2),
            N.LeakyReLU()
        )

        for i in range(self.depth):
            self.add_module(
                self.proxy(i),
                N.Sequential(
                    N.ConvTranspose1d(self.get_width(self.root_filter*2**i),self.get_width(self.root_filter*2**(i+1)),4,2,1),
                    N.LeakyReLU()
                )
            )

        self.out = N.Sequential(
            N.Conv1d(self.get_width(self.root_filter*2**self.depth),1,5,1,2),
            N.LeakyReLU()
        )



    def forward(self,x):


        x = self.input(x)

        for ind in range(self.depth):

            module = self.proxy[ind]
            x = module(x)

        return self.out(x)