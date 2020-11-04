import torch.nn as N

from src.config import LATENT_SIZE, DEPTH_PARAMETER
from src.layers.BottleneckLayer import BottleneckLayer
from src.layers.BottleneckSwapout import BottleneckSwapout
from src.layers.conv_blocks import ConvBlock,DeConvBlock


class Generator(N.Module):

    def __init__(self,output_size,latent_size = LATENT_SIZE,depth_parameter=DEPTH_PARAMETER):
        super().__init__()

        self._output_size = output_size
        self._latent_size = latent_size
        self._depth_parameter = depth_parameter

        self.init_channels = (self._latent_size * 16) // self._output_size // self._output_size
        self._dimension =  self._output_size//4

        self.conv1 = N.Sequential(
            ConvBlock(self.init_channels,16,relu=True),
            ConvBlock(16,32,relu=True),
            ConvBlock(32, 64,relu=True),
            ConvBlock(64, 128,relu=True),
            BottleneckSwapout(128, 128, relu=True),
            BottleneckSwapout(128, 128, relu=True),
            BottleneckSwapout(128, 128, relu=True),

        )

        self.conv2 = N.Sequential(
            DeConvBlock(128,64),
            BottleneckSwapout(64, 64, relu=True),
            BottleneckSwapout(64, 64, relu=True),
            BottleneckSwapout(64, 64, relu=True),

        )

        self.conv3 = N.Sequential(
            DeConvBlock(64, 32),
            BottleneckSwapout(32, 32, relu=True),
            BottleneckSwapout(32, 32, relu=True),
            BottleneckSwapout(32, 32, relu=True),
        )
        self.conv4 = N.Sequential(
            N.Conv2d(32,1,1),
            N.ReLU(),
            N.Flatten()
        )



    def forward(self, x):


        x = x.view(x.shape[0],self.init_channels,self._dimension,self._dimension)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)
