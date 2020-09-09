import torch
import torch.nn as N


from src.torch.utils import get_conv_block


class Generator(N.Module):

    def __init__(self,output_size):
        super().__init__()

        self._output_size = output_size

        self.l1 = N.Sequential(
            N.Linear(100,output_size*output_size//16),
            N.Dropout(),
            N.SELU()
        )
        self.upsample1 = N.Sequential(
            N.ConvTranspose2d(128,64,5,padding=2,stride=2,output_padding=(1,1)),
            N.LeakyReLU(),
            N.BatchNorm2d(64)
        )

        self.upsample2 = N.Sequential(
            N.ConvTranspose2d(64, 64, 5, padding=2, stride=2, output_padding=(1, 1)),
            N.LeakyReLU(),
            N.BatchNorm2d(64)
        )

        self.conv1 = N.Sequential(*get_conv_block(1,128))
        self.conv2 = N.Sequential(*get_conv_block(64,64))
        self.conv3 = N.Sequential(*get_conv_block(64,64))
        self.conv4 = N.Sequential(*get_conv_block(64,1))

    def forward(self, x):

        x = self.l1(x)
        x = x.view(x.shape[0],1,self._output_size//4,self._output_size//4)
        x = self.upsample1(self.conv1(x))
        x = self.upsample2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)

        return torch.sigmoid(x)

