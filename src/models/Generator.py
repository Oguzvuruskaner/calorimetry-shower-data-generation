import torch.nn as N

from src.config import LATENT_SIZE, DEPTH_PARAMETER
from src.models.Swapout import Swapout
from src.utils import get_conv_block,get_dense_block


class Generator(N.Module):

    def __init__(self,output_size,latent_size = LATENT_SIZE,depth_parameter=DEPTH_PARAMETER):
        super().__init__()

        self._output_size = output_size
        self._latent_size = latent_size
        self._depth_parameter = depth_parameter


        self.l1 = N.Sequential(*depth_parameter*[Swapout(get_dense_block(latent_size,latent_size))])
        self.l2 = get_dense_block(latent_size,output_size*output_size)

        self.conv1 = N.Sequential(*get_conv_block(16,16) , *get_conv_block(16,32))
        self.conv2 = N.Sequential(*depth_parameter*[Swapout(get_conv_block(32,32))])
        self.conv3 = N.Sequential(*depth_parameter*[Swapout(get_conv_block(32,32))])
        self.conv4 = N.Sequential(*depth_parameter*[Swapout(get_conv_block(32,32))])
        self.conv5 = N.Sequential(
            get_conv_block(32, 1),
            N.Flatten(),
            N.ReLU(),
        )

        self.upsample1 = N.Sequential(
            N.Upsample(scale_factor=2),
        )

        self.upsample2 = N.Sequential(
            N.Upsample(scale_factor=2),
        )


    def forward(self, x):



        x = self.l1(x)
        x = self.l2(x)
        x = x.view(-1,16,self._output_size//4,self._output_size//4)

        x = self.conv1(x)
        x = self.upsample1(self.conv2(x))
        x = self.upsample2(self.conv3(x))
        x = self.conv4(x)

        return self.conv5(x)

