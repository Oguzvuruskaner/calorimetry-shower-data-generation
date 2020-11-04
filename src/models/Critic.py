import torch
import torch.nn as N
from torch.nn.utils import spectral_norm

from src.config import DEPTH_PARAMETER
from src.layers.BottleneckLayer import BottleneckLayer
from src.layers.BottleneckSwapout import BottleneckSwapout
from src.layers.MinibatchDiscrimination import MinibatchDiscrimination
from src.layers.conv_blocks import ConvBlock, SpectralConvBlock


class Critic(N.Module):

    def __init__(self, input_dim: int,depth_parameter = DEPTH_PARAMETER):
        super().__init__()


        self._input_dim = input_dim
        self._depth_parameter = depth_parameter

        self.conv1 = N.Sequential(
            SpectralConvBlock(1, 8),
            SpectralConvBlock(8, 16),
            BottleneckLayer(16, 16,relu=False,spectral=True),
            ConvBlock(16,32),
            BottleneckLayer(32,32,relu=False,spectral=True),
        )

        self.conv2 = N.Sequential(
            SpectralConvBlock(32,64,5,2,2),
            BottleneckLayer(64, 64,relu=False,spectral=True),
            BottleneckLayer(64, 64,relu=False,spectral=True)
        )

        self.conv3 = N.Sequential(
            SpectralConvBlock(64, 128,5, 2, 2),
            BottleneckLayer(128, 128,relu=False,spectral=True),
            BottleneckLayer(128, 128,relu=False,spectral=True),
            N.AdaptiveAvgPool2d((1,1)),
            N.Flatten()
        )

        self.minibatch_discrimination = MinibatchDiscrimination(
            128,64,32
        )

        self.output = N.Sequential(
            spectral_norm(N.Linear(64 + 128, 1)),
            N.Tanh()
        )



    def forward(self, x,feature_matching = False):

        x = x.view(x.shape[0],1,self._input_dim,self._input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        O = self.minibatch_discrimination(x)
        x = torch.cat([O, x], 1)

        if feature_matching:
            return x
        else:
            return self.output(x)
