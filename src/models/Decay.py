import torch.nn as N
import torch

class Decay(N.Module):

    def __init__(self, out_size):

        super().__init__()
        self._out_size = out_size
        self.weight = N.Parameter(torch.empty((out_size,1)))

        N.init.uniform_(self.weight,1,10)

    def forward(self,x):

        x = N.functional.relu(x,inplace=False)
        return torch.exp (-torch.abs(self.weight) * x)


