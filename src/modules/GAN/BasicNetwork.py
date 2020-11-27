import torch.nn as N
from torch.nn.functional import dropout

from src.helpers.AttrProxy import AttrProxy
from src.modules.helpers.DropoutModule import DropoutModule
from src.modules.helpers.MaxWidth import MaxWidth


class BasicNetwork(N.Module,DropoutModule,MaxWidth):

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



    def forward(self,x):


        x = self.input(x)

        for ind in range(self.depth):

            x = dropout(
                x,
                p = self.dropout_rate * ind/self.depth,
                training=self.training
            )

            module = self.proxy[ind]

            x = dropout(
                module(x),
                p=self.dropout_rate * ind / self.depth,
                training=self.training
            )


        return self.out(x)