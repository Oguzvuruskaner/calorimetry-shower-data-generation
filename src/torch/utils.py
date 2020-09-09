from typing import Union

import torch
import torch.nn as N


get_conv_block = lambda in_channel,out_channel: (
    N.Conv2d(in_channel, out_channel, 5,padding=2),
    N.LeakyReLU(inplace=True),
    N.Conv2d(out_channel, out_channel, 5,padding=2),
    N.LeakyReLU(inplace=True),
    N.Conv2d(out_channel, out_channel, 5,padding=2),
    N.LeakyReLU(inplace=True),
    N.BatchNorm2d(out_channel),
)

def initialize(m:N.Module):

    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data,-0.05)
        N.init.zeros_(m.bias.data)

def wasserstein_loss(output,target):

    return torch.mean(-output*target)


def get_next_iter(dataloader):

    return next(iter(dataloader))


