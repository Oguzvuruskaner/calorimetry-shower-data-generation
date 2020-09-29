import torch.nn as N

from src.layers.BottleneckLayer import BottleneckLayer


class BottleneckSwapout(BottleneckLayer):

    def __init__(self, in_channels,out_channels,f_x_survival_prob = 0.7,x_survival_prob = 0.8,relu=False):
        super().__init__(in_channels,out_channels,relu = relu)

        self.f_x_dropout = N.Dropout(1-f_x_survival_prob)
        self.x_dropout = N.Dropout(1-x_survival_prob)


    def forward(self,x):
        f_x = super().forward(x)
        return self.x_dropout(x) + self.f_x_dropout(f_x)