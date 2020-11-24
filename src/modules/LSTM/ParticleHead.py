import torch.nn as N
import torch

class ParticleHead(N.Module):


    def forward(self,x):

        tanh = torch.tanh(x[:,:,:-1])
        sigmoid = torch.sigmoid(x[:,:,-1:])
        return torch.cat([tanh,sigmoid],dim=2)