import torch.nn as N

class ResidualLayer(N.Module):

    def __init__(self,inner_layer ):
        super().__init__()
        self.inner = inner_layer()


    def forward(self,x):

        #Assuming that output dimensions of inner layer are same as x.

        return x + self.inner(x)