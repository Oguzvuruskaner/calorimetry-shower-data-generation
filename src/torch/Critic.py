import torch
import torch.nn as N
import torch.nn.functional as F

from src.torch.MinibatchDiscrimination import MinibatchDiscrimination
from src.torch.utils import get_conv_block


class Critic(N.Module):

    def __init__(self, input_dim: int,number_of_labels):
        super().__init__()


        self._input_dim = input_dim
        self._number_of_labels = number_of_labels

        self.conv1 = N.Sequential(*get_conv_block(1,64),N.Conv2d(64,64,5,padding=2,stride=2),N.LeakyReLU(inplace=True),N.BatchNorm2d(64))
        self.conv2 = N.Sequential(*get_conv_block(64,64),N.Conv2d(64,64,5,padding=2,stride=2),N.LeakyReLU(inplace=True),N.BatchNorm2d(64))
        self.conv3 = N.Sequential(*get_conv_block(64,128), N.BatchNorm2d(128))

        self.l1 = N.Sequential(
            N.Flatten(),
            N.Linear(input_dim * input_dim * 128 // 16,128),
            N.BatchNorm1d(128)
        )

        self.minibatch_discrimination = MinibatchDiscrimination(
            128,64,32
        )

        self.real_fake = N.Sequential(
            N.Linear(64 + 128, 128),
            N.SELU(inplace=True),
            N.BatchNorm1d(128),
            N.Linear(128,1),
            N.Sigmoid()
        )

        self.label = N.Sequential(
            N.Linear(64+128,128),
            N.SELU(inplace=True),
            N.BatchNorm1d(128),
            N.Linear(128,number_of_labels),
            N.Softmax(dim=1)
        )




    def forward(self, x):

        x = x.view(x.shape[0],1,self._input_dim,self._input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.l1(x)
        O = self.minibatch_discrimination(x)
        x = torch.cat([O,x],1)

        return self.real_fake(x),self.label(x)
