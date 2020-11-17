from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.modules.LSTM import LSTM
from src.modules.SingleLabelDataset import SingleLabelDataset

import os


if __name__ == "__main__":


    model = LSTM()
    datamodule = SingleLabelDataset()
    datamodule.setup()
    callbacks = []
    loggers = [
        TensorBoardLogger(os.path.join("..","logs","lstm"))
    ]

    trainer = Trainer(loggers=loggers,gpus=1,auto_select_gpus=True)
    trainer.fit(model,datamodule.train_dataloader())