from random import choice

from pytorch_lightning import LightningDataModule
import os
import tables


from torch.utils.data import DataLoader,Dataset
import torch

# Data is jets filled with zeros if there are empty places.
from tqdm import tqdm


class JetDataset(Dataset):

    def __init__(self,filepath:str,max_jet_size = 1500,steps_per_epoch=64):

        self._file = tables.open_file(filepath,mode="r")
        self.max_jet_size = max_jet_size
        self.steps_per_epoch = steps_per_epoch
        self.data = []

        self.setup_data()

    def setup_data(self):
        data = self._file.root["data"]


        for jet in data:
            if len(jet)//4 <= self.max_jet_size:
                jet = torch.from_numpy(jet)
                self.data.append(jet.view(-1, 4))


        self._file.close()


    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, item):

        return choice(self.data)




class DataModule(LightningDataModule):

    def __init__(self,root_dir, dataset_label=1,batch_size=1, steps_per_epoch=300):

        super().__init__()
        self.dataset_label = dataset_label
        self.root_dir = root_dir
        self.train_data = None
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size


    def setup(self,*args, **kwargs):
        self.train_data = JetDataset(os.path.join(self.root_dir,"all.h5"))

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)

