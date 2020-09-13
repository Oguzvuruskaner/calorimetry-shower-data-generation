import torch
import torch.nn as N


from src.torch.utils import get_conv_block


class Generator(N.Module):

    def __init__(self,output_size,latent_size = 100,number_of_labels = 10):
        super().__init__()

        self._output_size = output_size
        self._latent_size = latent_size
        self._number_of_labels = number_of_labels

        self.embedding = N.Embedding(number_of_labels,self._latent_size)


        self.l1 = N.Sequential(
            N.Linear(self._latent_size,output_size*output_size//16),
            N.SELU(),
            N.BatchNorm1d(output_size*output_size//16)
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
        self.conv4 = N.Sequential(*get_conv_block(64,1),N.Flatten())

    def forward(self, z,label):

        embedding = self.embedding(label)
        embbeding = embedding.view(z.shape[0],-1)
        x = embbeding * z
        x = self.l1(x)
        x = x.view(x.shape[0],1,self._output_size//4,self._output_size//4)
        x = self.upsample1(self.conv1(x))
        x = self.upsample2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)

        return torch.sigmoid(x)

