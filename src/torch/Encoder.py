import torch
import torch.nn as N

from src.torch.utils import get_conv_block


class Encoder(N.Module):

    def __init__(self,input_dimension:int):

        super(Encoder, self).__init__()
        self._input_dimension = input_dimension


        self.conv1 = N.Sequential(
            *get_conv_block(1,64),
            N.Conv2d(64,64,5,2,2)
        )
        self.conv2 = N.Sequential(
            *get_conv_block(64, 64),
            N.Conv2d(64, 128, 5, 2, 2)
        )
        self.conv3 = N.Sequential(
            *get_conv_block(128,128)
        )
        self.mean = N.Sequential(
            N.Flatten(),
            N.Linear(128 * input_dimension * input_dimension // 16, input_dimension*input_dimension // 16),
        )
        self.var = N.Sequential(
            N.Flatten(),
            N.Linear(128 * input_dimension * input_dimension // 16, input_dimension*input_dimension // 16),
        )

    def forward(self,x ):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)


        return self.mean(x),self.var(x)
