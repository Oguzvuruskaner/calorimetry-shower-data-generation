import numpy as np
from tqdm import tqdm

from src.datasets import DATASETS, entries
import uproot4 as uproot

def analyze_dataset(entry_directory,histogram_bins = 100,step_size = 128):

    NUMBER_OF_ENTRIES = entry_directory.num_entries
    MAX = np.array(4*[-float("inf")])
    MIN = np.array(4*[float("inf")])

    MEAN = np.zeros((4,1))
    TOTAL_PARTICLES = 0
    # In first iteration mean value , max and min values are going to be calculated.

    for ind,(x_batch,y_batch,z_batch,e_batch) in tqdm(enumerate(zip(
            entry_directory["hit_x"].iterate(step_size=step_size,library="np"),
            entry_directory["hit_y"].iterate(step_size=step_size,library="np"),
            entry_directory["hit_z"].iterate(step_size=step_size,library="np"),
            entry_directory["hit_e"].iterate(step_size=step_size,library="np"),
    ))):

        x_batch = x_batch["hit_x"]
        y_batch = y_batch["hit_y"]
        z_batch = z_batch["hit_z"]
        e_batch = e_batch["hit_e"]


        for x,y,z,e in zip(x_batch,y_batch,z_batch,e_batch):


            if MIN[0] > x.min():
                MIN[0] = x.min()
            if MAX[0] < x.max():
                MAX[0] = x.max()

            if MIN[1] > y.min():
                MIN[1] = y.min()
            if MAX[1] < y.max():
                MAX[1] = y.max()

            if MIN[2] > z.min():
                MIN[2] = z.min()
            if MAX[2] < z.max():
                MAX[2] = z.max()

            if MIN[3] > e.min():
                MIN[3] = e.min()
            if MAX[3] < e.max():
                MAX[3] = e.max()

            if TOTAL_PARTICLES == 0:
                TOTAL_PARTICLES += len(x)
                MEAN = np.array([
                    x.mean(),
                    y.mean(),
                    z.mean(),
                    e.mean()
                ])
            else:
                MEAN += [
                    x.mean(),
                    y.mean(),
                    z.mean(),
                    e.mean()
                ]
                TOTAL_PARTICLES += len(x)


    MEAN /= NUMBER_OF_ENTRIES

    get_x_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[0],MAX[0]))[0]
    get_y_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[1],MAX[1]))[0]
    get_z_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[2],MAX[2]))[0]
    get_e_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[3],MAX[3]))[0]

    hists = np.array(4 * [np.zeros(histogram_bins)])

    VARIANCE = np.array(4*[.0])

    for ind,(x_batch,y_batch,z_batch,e_batch) in tqdm(enumerate(zip(
            entry_directory["hit_x"].iterate(step_size=step_size,library="np"),
            entry_directory["hit_y"].iterate(step_size=step_size,library="np"),
            entry_directory["hit_z"].iterate(step_size=step_size,library="np"),
            entry_directory["hit_e"].iterate(step_size=step_size,library="np"),
    ))):

        x_batch = x_batch["hit_x"]
        y_batch = y_batch["hit_y"]
        z_batch = z_batch["hit_z"]
        e_batch = e_batch["hit_e"]


        for x,y,z,e in zip(x_batch,y_batch,z_batch,e_batch):


            hists[0] += get_x_hist(x)
            hists[1] += get_y_hist(y)
            hists[2] += get_z_hist(z)
            hists[3] += get_e_hist(e)

            VARIANCE[0] += np.sum((x-MEAN[0]))**2/len(x)
            VARIANCE[1] += np.sum((y-MEAN[1]))**2/len(x)
            VARIANCE[2] += np.sum((z-MEAN[2]))**2/len(x)
            VARIANCE[3] += np.sum((e-MEAN[3]))**2/len(x)


    VARIANCE /=60000
    STD = VARIANCE**.5


def main():

    for GeV in DATASETS.keys():



        for dataset in DATASETS[GeV]:

            entry_strat = dataset.get("entries",None) or entries
            dataset_path = dataset["path"]


            with uproot.open(dataset_path) as root:

                entry_directory = entry_strat(root)
                analyze_dataset(entry_directory)