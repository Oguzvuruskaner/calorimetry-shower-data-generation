import torch
import torch.nn as N

from src.torch.utils import get_conv_block


class Critic(N.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.conv1 = N.Sequential(*get_conv_block(1,64),N.Conv2d(64,64,5,padding=2,stride=2),N.LeakyReLU(inplace=True),N.BatchNorm2d(64))
        self.conv2 = N.Sequential(*get_conv_block(64,64),N.Conv2d(64,64,5,padding=2,stride=2),N.LeakyReLU(inplace=True),N.BatchNorm2d(64))
        self.conv3 = N.Sequential(*get_conv_block(64,128), N.BatchNorm2d(128))

        self.output = N.Sequential(
            N.Flatten(),
            N.Linear(input_dim * input_dim * 128 // 16, 1)
        )

    def get_middle_layers(self):

        return [
            self.conv1,
            N.Sequential(self.conv1,self.conv2),
            N.Sequential(self.conv1, self.conv2,self.conv3),
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output(x)

        return torch.sigmoid(x)
