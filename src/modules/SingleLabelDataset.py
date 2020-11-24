from pytorch_lightning import LightningDataModule
import os
import tables

from torch.utils.data import DataLoader,Dataset
import torch


class JetDataset(Dataset):

    def __init__(self,filepath:str,max_jet_size = 96):

        self._file = tables.open_file(filepath,mode="r")
        self.max_jet_size = max_jet_size

        self.setup_data()

    def setup_data(self):
        data = self._file.root["data"]
        self.data = torch.zeros((len(data),1, self.max_jet_size, 4))

        for ind,jet in enumerate(data):
            jet = jet.reshape(-1,4)
            self.data[ind] = torch.from_numpy(jet[:self.max_jet_size])

        self._file.close()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item]


class SingleLabelDataset(LightningDataModule):

    def __init__(self, dataset_label=1,batch_size=1024, steps_per_epoch=300):

        super().__init__()
        self.dataset_label = dataset_label
        self.root_dir = os.path.join("..","data","particle_dataset","{}".format(dataset_label))
        self.train_data = None
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size


    def setup(self):
        self.train_data = JetDataset(os.path.join(self.root_dir,"all.h5"))

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True)

