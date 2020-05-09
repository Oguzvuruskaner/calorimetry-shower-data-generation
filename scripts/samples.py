from tqdm import tqdm
import numpy as np
from os import path
from config import __DATASETS_WITH_OUTLIERS__


def get_samples(sample_size=10**6):

    for dataset_name in tqdm(__DATASETS_WITH_OUTLIERS__):

        data = np.load(path.join("npy", "{}.npy".format(dataset_name)), allow_pickle=True)

        if len(data.shape) <= 2:
            increment = data.shape[0]//sample_size
            if increment == 0:
                raise Exception("Size of sample : {} is bigger than dataset size {}.".format(sample_size,data.shape[0]))

            else:
                #Using systematic sampling to keep array distribution still.
                data = np.sort(data)
                np.save(path.join("npy", "{}_{}.npy".format(dataset_name, sample_size)), data[::increment])

