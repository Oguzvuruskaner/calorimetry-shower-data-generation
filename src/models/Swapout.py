import torch.nn as N

class Swapout(N.Module):

    def __init__(self, inner_layer : N.Module,f_x_survival_prob = 0.7,x_survival_prob = 0.8):
        super().__init__()

        self._inner_layer = inner_layer
        self.f_x_dropout = N.Dropout(1-f_x_survival_prob)
        self.x_dropout = N.Dropout(1-x_survival_prob)


    def forward(self,x):

        f_x = self._inner_layer(x)
        return self.x_dropout(x) + self.f_x_dropout(f_x)