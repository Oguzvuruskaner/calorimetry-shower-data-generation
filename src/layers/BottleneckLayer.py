import torch.nn as N

class BottleneckLayer(N.Module):

    def __init__(self,in_channels,out_channels,relu = True):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self.conv = N.Sequential(
            N.Conv2d(in_channels,in_channels,1),
            N.BatchNorm2d(in_channels),
            N.PReLU() if not relu else N.ReLU(),
            N.Conv2d(in_channels,in_channels,3,padding=1),
            N.BatchNorm2d(in_channels),
            N.PReLU() if not relu else N.ReLU(),
            N.Conv2d(in_channels, out_channels, 1),
            N.BatchNorm2d(out_channels),
            N.PReLU() if not relu else N.ReLU(),
        )


    def get_inner_layer(self) -> N.Sequential:

        return self.conv

    def forward(self,x):

        #Assuming that output dimensions of inner layer are same as x.

        return x + self.conv(x)