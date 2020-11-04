import torch
import torch.nn as N
import os
from random import random

def create_or_cleanup(folder) -> object:
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
        N.init.kaiming_normal_(m.weight.data)
        N.init.uniform_(m.bias.data,-0.01,0.01)

def critic_init(m:N.Module):

    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data)
        N.init.uniform_(m.bias.data,-0.01,0.01)




def decay_dropout_rate(model, decay=0.98):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            if child.p < 0.01:
                child.p = 0
            else:
                child.p = child.p * decay
        decay_dropout_rate(child, decay=decay)


def iterate_array(arr,batch_size):

    NUMBER_OF_ENTRIES = arr.num_entries
    for i in range(0,NUMBER_OF_ENTRIES//batch_size):
        yield arr.array(entry_start=i*batch_size,entry_stop=(i+1)*batch_size,library="np",array_cache=None)



def bernoulli(prob):
    return prob > random()