import torch
import torch.nn as N

from src.config import DEPTH_PARAMETER
from src.layers.BottleneckSwapout import BottleneckSwapout
from src.layers.MinibatchDiscrimination import MinibatchDiscrimination
from src.layers.conv_blocks import ConvBlock

class Critic(N.Module):

    def __init__(self, input_dim: int,depth_parameter = DEPTH_PARAMETER):
        super().__init__()


        self._input_dim = input_dim
        self._depth_parameter = depth_parameter

        self.conv1 = N.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            BottleneckSwapout(16, 16),
            ConvBlock(16,32),
            BottleneckSwapout(32,32),
        )

        self.conv2 = N.Sequential(
            ConvBlock(32,64,5,2,2),
            BottleneckSwapout(64, 64),
            BottleneckSwapout(64, 64)
        )

        self.conv3 = N.Sequential(
            ConvBlock(64, 128,5, 2, 2),
            BottleneckSwapout(128, 128),
            BottleneckSwapout(128, 128),
            N.AdaptiveAvgPool2d((1,1)),
            N.Flatten()
        )

        self.minibatch_discrimination = MinibatchDiscrimination(
            128,64,32
        )

        self.output = N.Sequential(
            N.Linear(64 + 128, 1),
            N.Tanh()
        )



    def forward(self, x,feature_matching = False):

        x = x.view(x.shape[0],1,self._input_dim,self._input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        O = self.minibatch_discrimination(x)

        if feature_matching:
            x = torch.cat([O,x],1)
            return x
        else:
            return self.output(x)
