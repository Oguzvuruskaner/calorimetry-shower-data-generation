from src.modules.GAN.Lightning import GANLightning
from src.modules.GAN.Dataset import DataModule

from pytorch_lightning import Trainer

import os

if __name__ == "__main__":


    DATA_ROOT = os.path.join("..","..","..","data","particle_dataset","1")


    datamodule = DataModule(DATA_ROOT)
    model = GANLightning()

    callbacks = [

    ]

    trainer = Trainer(gpus=1,auto_select_gpus=True,callbacks=callbacks)

    trainer.fit(model,datamodule=datamodule)

