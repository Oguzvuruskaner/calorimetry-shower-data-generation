from tqdm import trange

from src.datasets.Dataset import Dataset
import uproot
import numpy as np
from src.config import HIT_R_MAX, HIT_R_MIN, HIT_Z_MAX, HIT_Z_MIN, DIMENSION
from src.decorators.Builder import BuilderMethod


class JetImageDataset(Dataset):

    def __init__(self,
                 root_directories: [str] = (),
                 store_path=None,
                 dimension=DIMENSION
                 ):

        super().__init__()
        self._root_directories = root_directories
        self._store_path = store_path
        self._dimension = dimension
        self._np_path = None

    @BuilderMethod
    def set_store_path(self, store_path: str) -> "JetImageDataset":

        self._store_path = store_path
        return self

    @BuilderMethod
    def set_np_path(self, np_path: str) -> "JetImageDataset":
        self._np_path = np_path
        return self

    @BuilderMethod
    def store(self) -> "JetImageDataset":

        np.save(
            self._store_path,
            self._data,
            allow_pickle=True
        )

    @BuilderMethod
    def obtain(self, from_npy=False) -> "JetImageDataset":

        if from_npy == False:
            self._obtain_from_root()
        else:
            self._obtain_from_npy()

    def _obtain_from_npy(self):

        self._data = np.load(self._np_path, allow_pickle=True)

    def _obtain_from_root(self):

        root_paths = Dataset.get_root_files_in_multiple_directories(self._root_directories)
        self._bootstrap_data_array(root_paths)

        current_element = 0

        for root_path in root_paths:
            with uproot.open(root_path) as root:

                hit_x = np.array(root[b"showers"][b"hit_x"].array())
                hit_y = np.array(root[b"showers"][b"hit_y"].array())
                hit_z = np.array(root[b"showers"][b"hit_z"].array())
                hit_e = np.array(root[b"showers"][b"hit_e"].array())

                for i in trange(len(hit_x)):

                    tmp_jet = np.zeros((len(hit_x[i]), 3))

                    tmp_jet[:, 0] = np.sqrt(hit_x[i] * hit_x[i] + hit_y[i] * hit_y[i])
                    tmp_jet[:, 1] = hit_z[i]
                    tmp_jet[:, 2] = hit_e[i]

                    tmp_jet[:, 0] = np.floor((tmp_jet[:, 0] - HIT_R_MIN) / (HIT_R_MAX - HIT_R_MIN) * DIMENSION)
                    tmp_jet[:, 1] = np.floor((tmp_jet[:, 1] - HIT_Z_MIN) / (HIT_Z_MAX - HIT_Z_MIN) * DIMENSION)

                    for r, z, e in tmp_jet:
                        self._data[current_element, int(z), int(r)] += e

                    current_element += 1

    def _bootstrap_data_array(self, root_paths: [str]):

        total_jets = 0

        for root_path in root_paths:
            with uproot.open(root_path) as root:
                total_jets += len(root[b"showers"][b"hit_x"].array())

        self._data = np.zeros((total_jets, self._dimension, self._dimension, 1))




