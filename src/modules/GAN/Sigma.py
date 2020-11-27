import torch.nn as N
import math
from src.modules.GAN.Up import Up


class Sigma(Up):

    def __init__(self, *args, **kwargs):


        super().__init__(*args,**kwargs)


        self.out = N.Sequential(
            N.Conv1d(self.root_filter, 1, 5, 1, 2),
            N.Sigmoid()
        )
