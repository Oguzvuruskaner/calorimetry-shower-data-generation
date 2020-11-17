from pytorch_lightning import LightningDataModule
import os
import tables

from torch.utils.data import DataLoader,Dataset



class JetDataset(Dataset):

    def __init__(self,filepath:str):

        self._file = tables.open_file(filepath,mode="r")
        self.data = self._file.root["data"]

    def __del__(self):
        self._file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item]


class SingleLabelDataset(LightningDataModule):

    def __init__(self,dataset_label = 1):

        self.dataset_label = dataset_label
        self.root_dir = os.path.join("..","data","particle_dataset","{}".format(dataset_label))
        self.train_data = None


    def setup(self):
        self.train_data = JetDataset(os.path.join(self.root_dir,"all.h5"))

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_data,batch_size=1,shuffle=True)

