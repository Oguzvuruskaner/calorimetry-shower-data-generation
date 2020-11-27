from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.callbacks.LogGenerathings import LogGenerathings
from src.modules.GAN.Lightning import GANLightning
from src.modules.GAN.Dataset import DataModule

from pytorch_lightning import Trainer

import os

if __name__ == "__main__":


    DATA_ROOT = os.path.join("..","..","..","data","particle_dataset","1")
    LOG_DIR = os.path.join("..","..","..","logs","generathings")
    MODEL_DIR = os.path.join("..","..","..","models","gan")

    datamodule = DataModule(DATA_ROOT,steps_per_epoch=32,max_jet_size=1500)
    model = GANLightning()

    callbacks = [
        LogGenerathings(LOG_DIR,600),
        ModelCheckpoint(dirpath=MODEL_DIR,monitor="gen_loss",save_top_k=3),
        EarlyStopping(monitor="gen_loss",mode="min",patience=10)
    ]

    trainer = Trainer(gpus=1,auto_select_gpus=True,callbacks=callbacks)

    trainer.fit(model,datamodule=datamodule)

