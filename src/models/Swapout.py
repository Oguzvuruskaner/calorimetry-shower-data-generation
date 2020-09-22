import torch
import torch.nn as N
from torch.distributions.bernoulli import Bernoulli

class Swapout(N.Module):

    def __init__(self, inner_layer : N.Module,f_x_survival_prob = 0.7,x_survival_prob = 0.8):
        super().__init__()

        self._inner_layer = inner_layer
        self.f_x_bernoulli = Bernoulli(f_x_survival_prob)
        self.x_bernoulli = Bernoulli(x_survival_prob)


    def forward(self,x):

        if self.training:
            x_survival_matrix = self.x_bernoulli.sample(x.shape).to(x.device)
            f_x_survival_matrix = self.f_x_bernoulli.sample(x.shape).to(x.device)

            return x * x_survival_matrix + f_x_survival_matrix * self._inner_layer(x)

        else:
            return x + self._inner_layer(x)