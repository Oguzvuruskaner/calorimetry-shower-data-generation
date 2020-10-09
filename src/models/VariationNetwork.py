import torch.nn as N
import torch
from torch.nn.functional import dropout2d
from torch.nn.utils import spectral_norm

from src.config import STATE_SIZE
from src.layers.AttrProxy import AttrProxy


class VariationNetwork(N.Module):

    def __init__(self,input_size,number_of_labels, state_size=STATE_SIZE, depth=18):

        super().__init__()

        self.depth = depth
        self.input_size = input_size
        self.number_of_labels = number_of_labels
        self.state_size = state_size

        self.proxy = AttrProxy(self,"l_")

        self.add_module(self.proxy("inp"), N.Sequential(
            spectral_norm(N.Linear(self.input_size + self.number_of_labels, self.state_size)),
            N.SELU()
        ))

        self.embedding = N.Embedding(self.number_of_labels, self.number_of_labels)
        self.embedding.weight.data = torch.eye(self.number_of_labels)
        self.embedding.weight.requires_grad_(False)

        for i in range(self.depth):
            self.add_module(
                self.proxy("{}".format(i)),
                N.Sequential(
                    spectral_norm(N.Linear(self.state_size, self.state_size)),
                    N.SELU()
                )
            )

        self.add_module(self.proxy("out"),
            N.Sequential(
                spectral_norm(N.Linear(self.state_size, self.state_size)),
                N.ReLU()
            )
        )


        self.inference_state = {
            "label":None
        }

    def set_label(self,label:int):
        self.inference_state["label"] = label

    def forward(self,x):

        one_hot_vector = self.embedding(self.inference_state["label"]).view(1,-1)
        x = self.proxy["inp"](torch.cat([x,one_hot_vector],dim=1))

        for ind in range(self.depth):
            module = self.proxy["{}".format(ind)]
            x = x + module(x)


        return self.proxy["out"](x)