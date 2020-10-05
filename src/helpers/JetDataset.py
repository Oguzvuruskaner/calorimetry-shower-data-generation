from torch.utils.data import Dataset
import tables

class JetDataset(Dataset):

    def __init__(self,filepath:str):

        self._file = tables.open_file(filepath,mode="r")
        self.data = self._file.root["data"]
        self.labels = self._file.root["labels"]

    def __del__(self):
        self._file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        return self.data[item],self.labels[item]
