from src.callbacks.LogGenerathings import LogGenerathings
from src.modules.GAN.Lightning import GANLightning
from src.modules.GAN.Dataset import DataModule

from pytorch_lightning import Trainer

import os

if __name__ == "__main__":


    DATA_ROOT = os.path.join("..","..","..","data","particle_dataset","1")
    LOG_DIR = os.path.join("..","..","..","logs","generathings")


    datamodule = DataModule(DATA_ROOT,steps_per_epoch=1,max_jet_size=500)
    model = GANLightning()

    callbacks = [
        LogGenerathings(LOG_DIR,600)
    ]

    trainer = Trainer(gpus=1,auto_select_gpus=True,callbacks=callbacks)

    trainer.fit(model,datamodule=datamodule)

