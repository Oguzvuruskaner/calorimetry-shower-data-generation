import torch.nn as N
from torch.nn.utils import spectral_norm

ConvBlock = lambda in_channels,out_channels,kernel_size=5,stride=1,padding=2,relu=False : N.Sequential(
    N.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    N.BatchNorm2d(out_channels),
    N.PReLU() if not relu else N.ReLU()
)

DeConvBlock = lambda in_channels,out_channels,kernel_size=2,stride=2,padding=0,relu=False : N.Sequential(
    N.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
    N.BatchNorm2d(out_channels),
    N.PReLU() if not relu else N.ReLU()
)

SpectralDeConv = lambda in_channels,out_channels,kernel_size=2,stride=2,padding=0,relu=False : N.Sequential(
    spectral_norm(N.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)),
    N.PReLU() if not relu else N.ReLU()
)

SpectralConvBlock = lambda in_channels,out_channels,kernel_size=5,stride=1,padding=2,relu=False : N.Sequential(
    spectral_norm(N.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)),
    N.PReLU() if not relu else N.ReLU()
)