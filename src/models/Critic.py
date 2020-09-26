import torch
import torch.nn as N

from src.config import DEPTH_PARAMETER
from src.models.MinibatchDiscrimination import MinibatchDiscrimination
from src.models.ResidualLayer import ResidualLayer
from src.models.Swapout import Swapout
from src.utils import get_conv_block, get_dense_block


class Critic(N.Module):

    def __init__(self, input_dim: int,depth_parameter = DEPTH_PARAMETER):
        super().__init__()


        self._input_dim = input_dim
        self._depth_parameter = depth_parameter

        self.conv1 = N.Sequential(get_conv_block(1, 16), get_conv_block(16, 32))
        self.conv2 = N.Sequential(*depth_parameter * [Swapout(get_conv_block(32, 32))])
        self.conv3 = N.Sequential(*depth_parameter * [Swapout(get_conv_block(32, 32))])
        self.conv4 = N.Sequential(*depth_parameter * [Swapout(get_conv_block(32, 32))])

        self.downsample1 = N.Sequential(
            N.Conv2d(32,32,5,padding=2,stride=2),
            N.BatchNorm2d(32),
            N.LeakyReLU()
        )

        self.downsample2 = N.Sequential(
            N.Conv2d(32, 32, 5, padding=2, stride=2),
            N.BatchNorm2d(32),
            N.LeakyReLU()
        )

        self.l1 = N.Sequential(
            N.Flatten(),
            N.Linear(input_dim * input_dim * 2,128),
            N.BatchNorm1d(128),
            N.LeakyReLU(),
            *depth_parameter * [Swapout(get_dense_block(128, 128))]
        )

        self.minibatch_discrimination = MinibatchDiscrimination(
            128,64,32
        )

        self.real_fake = N.Sequential(
            N.Linear(64 + 128, 128),
            N.LeakyReLU(),
            N.Linear(128,1),
            N.Tanh()
        )



    def forward(self, x):

        x = x.view(x.shape[0],1,self._input_dim,self._input_dim)
        x = self.conv1(x)
        x = self.downsample1(self.conv2(x))
        x = self.downsample2(self.conv3(x))
        x = self.conv4(x)
        x = self.l1(x)
        O = self.minibatch_discrimination(x)
        x = torch.cat([O,x],1)

        return self.real_fake(x)
