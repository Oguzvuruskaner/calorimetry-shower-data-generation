import torch.nn as N
import torch
from torch.nn.functional import dropout2d
from torch.nn.utils import spectral_norm

from src.config import STATE_SIZE
from src.layers.AttrProxy import AttrProxy


class EvaluationNetwork(N.Module):

    def __init__(self,number_of_labels, state_size=STATE_SIZE, depth=12,dropout_rate=0.4):

        super().__init__()

        self.depth = depth
        self.dropout_rate = dropout_rate
        self.number_of_labels = number_of_labels
        self.state_size = state_size

        self.l = AttrProxy(self,"l_")


        self.add_module(self.l("inp"),
            N.Sequential(
                spectral_norm(N.Linear(self.state_size + self.number_of_labels, self.state_size)),
                N.SELU())
        )

        self.embedding = N.Embedding(self.number_of_labels, self.number_of_labels)
        self.embedding.weight.data = torch.eye(self.number_of_labels)
        self.embedding.weight.requires_grad_(False)

        for ind in range(self.depth):
            self.add_module(
                self.l("{}".format(ind)),
                N.Sequential(
                    spectral_norm(N.Linear(self.state_size, self.state_size)),
                    N.SELU()
                )
            )

        self.add_module(self.l("out"),
            N.Sequential(
                spectral_norm(N.Linear(self.state_size, 1)),
                N.Sigmoid()
            )
        )

        self.inference_state = {
            "label":None
        }

    def decay_dropout_rate(self,alpha):
        self.dropout_rate = self.dropout_rate * alpha
        if self.dropout_rate < 0.01:
            self.dropout_rate = 0

    def set_label(self,label:int):
        self.inference_state["label"] = label

    def forward(self,state):

        one_hot_vector = self.embedding(self.inference_state["label"]).view(1,-1)
        x = self.l["inp"](torch.cat([state,one_hot_vector],dim=1))

        for ind in range(self.depth):
            module = self.l["{}".format(ind)]
            x = dropout2d(x,ind / self.depth * self.dropout_rate) +\
                dropout2d(module(x), (ind+1) / self.depth * self.dropout_rate)


        return self.l["out"](x)