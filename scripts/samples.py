from tqdm import tqdm
import numpy as np
from os import path
from config import __DATASETS_WITH_OUTLIERS__


def get_samples(sample_size=10**6):

    for dataset_name in tqdm(__DATASETS_WITH_OUTLIERS__):

        data = np.load(path.join("npy", "{}.npy".format(dataset_name)), allow_pickle=True)
        increment = data.shape[0]//sample_size
        np.save(path.join("npy", "{}_{}.npy".format(dataset_name, sample_size)), data[::increment])

