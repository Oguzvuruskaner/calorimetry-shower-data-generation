import torch
import torch.nn as N
import os

def create_or_cleanup(folder):
    if os.path.exists(folder):
        for basename in os.listdir(folder):
            os.unlink(os.path.join(folder,basename))
    else:
        os.mkdir(folder)

get_conv_block = lambda in_channel,out_channel: N.Sequential(
    N.Conv2d(in_channel, out_channel, 5,padding=2),
    N.BatchNorm2d(out_channel),
    N.LeakyReLU(),
    N.Conv2d(out_channel, out_channel, 5,padding=2),
    N.BatchNorm2d(out_channel),
    N.LeakyReLU(),
)

get_dense_block = lambda input_width,out_width:N.Sequential(
    N.Linear(input_width,input_width),
    N.BatchNorm1d(input_width),
    N.LeakyReLU(),
    N.Linear(input_width,out_width),
    N.BatchNorm1d(out_width),
    N.LeakyReLU()
)

def generator_init(m:N.Module):

    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.constant_(m.weight.data,0)
        N.init.constant_(m.bias.data,0)

def critic_init(m:N.Module):

    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data)
        N.init.constant_(m.bias.data,0)




