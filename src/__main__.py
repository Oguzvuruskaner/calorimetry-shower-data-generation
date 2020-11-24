from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.callbacks.LogGenerathings import LogGenerathings
from src.modules.LSTM.Lightning import LSTMLightning
from src.modules.SingleLabelDataset import SingleLabelDataset

import os


if __name__ == "__main__":


    LOG_DIR = os.path.join("..","logs","generathings")

    model = LSTMLightning()
    datamodule = SingleLabelDataset()
    datamodule.setup()

    callbacks = [
        ModelCheckpoint(os.path.join("..","models","lstm"),monitor="train_loss",save_top_k=3,mode="min"),
        LearningRateMonitor(),
        EarlyStopping(patience = 10),
        LogGenerathings(LOG_DIR,64),
    ]
    loggers = [
        TensorBoardLogger(os.path.join("..","logs","lstm")),
    ]

    trainer = Trainer(logger=loggers,gpus=1,max_epochs=10000,callbacks=callbacks,auto_select_gpus=True)
    trainer.fit(model,datamodule.train_dataloader())


