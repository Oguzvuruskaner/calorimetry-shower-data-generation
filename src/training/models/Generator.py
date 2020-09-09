import chainer

from chainer import links as L
from chainer import functions as F
from chainer import optimizers as O

import numpy as np
from src.config import LATENT_SIZE,DIMENSION


class Generator(chainer.Chain):

    def __init__(self,latent_size = LATENT_SIZE,output_width=DIMENSION*DIMENSION,weight_scale=0.02) -> None:

        super().__init__()
        self._latent_size = latent_size
        self._output_width = output_width
        self._init_scale = weight_scale


        with self.init_scope():

            w = chainer.initializers.Normal(self._init_scale)

            self.l0 = L.Linear(self._latent_size,output_width*output_width//64,initialW=w)

            self.dc0 = L.Deconvolution2D()


