import torch
import torch.nn as N

from src.config import STATE_SIZE
from src.layers.AttrProxy import AttrProxy

class CustomActivation(N.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x : torch.Tensor):
        return torch.cat([x[:3].tanh(),x[3].relu().view(1)])

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Subnetwork(N.Module):

    def __init__(self, inp_size=STATE_SIZE, out_size=STATE_SIZE,depth=4,Activation : N.Module = N.Sigmoid):
        super().__init__()

        self.inp_size = inp_size
        self.out_size = out_size
        self.depth = depth
        self.proxy = AttrProxy(self,"l_")

        self.add_module(
            self.proxy("inp"),
            N.Sequential(
                N.Linear(inp_size,inp_size),
                N.LeakyReLU()
            )
        )

        for ind in range(depth):
            self.add_module(
                self.proxy("{}".format(ind)),
                N.Sequential(
                    N.Linear(inp_size,inp_size),
                    N.LeakyReLU()
                )
            )
            self.add_module(
                self.proxy("{}_norm".format(ind)),
                N.LayerNorm(inp_size)
            )

        self.add_module(
            self.proxy("out"),
            N.Sequential(
                N.Linear(inp_size,out_size),
                Activation()
            )
        )

    def forward(self,x):

        x = self.proxy["inp"](x)

        for ind in range(self.depth):
            module = self.proxy["{}".format(ind)]
            x = self.proxy["{}_norm".format(ind)](module(x) + x)

        return self.proxy["out"](x)


class RNN(N.Module):

    def __init__(self, state_size = STATE_SIZE):
        super().__init__()

        self.state_size = state_size

        self.forget_gate = Subnetwork(state_size,state_size)
        self.update_gate = Subnetwork(state_size,state_size)
        self.output_gate = Subnetwork(state_size,state_size)

        self.state_update = Subnetwork(state_size,state_size,Activation=N.ReLU)
        self.output_update = Subnetwork(state_size,state_size,Activation=N.Tanh)

        self.particle = Subnetwork(2*state_size,4,Activation=CustomActivation)


    def init_forget_gate(self,init_fn):
        self.forget_gate.apply(init_fn)

    def init_update_gate(self,init_fn):
        self.update_gate.apply(init_fn)

    def init_output_gate(self,init_fn):
        self.output_gate.apply(init_fn)

    def init_gates(self,init_fn):
        self.update_gate.apply(init_fn)
        self.output_gate.apply(init_fn)
        self.forget_gate.apply(init_fn)


    def init_networks(self,init_fn):
        self.state_update.apply(init_fn)
        self.output_update.apply(init_fn)
        self.particle.apply(init_fn)

    def forward(self,c,h):

        particle = self.particle(torch.cat([c,h]))

        c = c * self.forget_gate(h)
        c += self.state_update(h) * self.update_gate(h)

        h += self.output_gate(h) * self.output_update(c)


        return particle,c,h
