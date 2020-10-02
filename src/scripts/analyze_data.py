import os

import numpy as np
from tqdm import tqdm

from src.datasets import DATASETS, entries
import uproot4 as uproot
import matplotlib.pyplot as plt

FEATURES = [
    "hit_x",
    "hit_y",
    "hit_z",
    "hit_e"
]
UNIT_OF_MEASURES = [
    "cm",
    "cm",
    "cm",
    "GeV"
]

def iterate_array(arr,batch_size):

    NUMBER_OF_ENTRIES = arr.num_entries
    for i in range(0,NUMBER_OF_ENTRIES//batch_size):
        yield arr.array(entry_start=i*batch_size,entry_stop=(i+1)*batch_size,library="np",array_cache=None)


def show_stats(analysis_dict:dict,index:int):

    tmp = ""
    tmp += "Total Entities : {} \n".format(analysis_dict["total_points"])
    tmp += "Mean : {0:10.3f} \n".format(analysis_dict["mean"][index])
    tmp += "Std : {0:10.3f} \n".format(analysis_dict["std"][index])
    tmp += "Variance : {0:10.3f}\n".format(analysis_dict["variance"][index])
    tmp += "Min : {0:10.3f}\n".format(analysis_dict["min"][index])
    tmp += "Max : {0:10.3f}\n".format(analysis_dict["max"][index])
    return tmp

def analyze_dataset(entry_directory,histogram_bins = 100,batch_size = 1000):

    NUMBER_OF_ENTRIES = entry_directory.num_entries
    MAX = np.array(4*[-float("inf")],dtype=np.float64)
    MIN = np.array(4*[float("inf")],dtype=np.float64)

    MEAN = np.zeros((4,1),dtype=np.float64)
    TOTAL_PARTICLES = 0
    # In first iteration mean value , max and min values are going to be calculated.

    for ind,(x_batch,y_batch,z_batch,e_batch) in tqdm(enumerate(zip(
            iterate_array(entry_directory["hit_x"],batch_size=batch_size),
            iterate_array(entry_directory["hit_y"],batch_size=batch_size),
            iterate_array(entry_directory["hit_z"],batch_size=batch_size),
            iterate_array(entry_directory["hit_e"],batch_size=batch_size)
    ))):


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

    get_x_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[0],MAX[0]))[0].reshape(histogram_bins,1)
    get_y_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[1],MAX[1]))[0].reshape(histogram_bins,1)
    get_z_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[2],MAX[2]))[0].reshape(histogram_bins,1)
    get_e_hist = lambda data : np.histogram(data,bins=histogram_bins,range=(MIN[3],MAX[3]))[0].reshape(histogram_bins,1)

    hists = np.array(4 * [np.zeros((histogram_bins,1))],dtype=np.int64)

    VARIANCE = np.array(4*[.0],dtype=np.float64)

    for ind,(x_batch,y_batch,z_batch,e_batch) in tqdm(enumerate(zip(
            iterate_array(entry_directory["hit_x"],batch_size=batch_size),
            iterate_array(entry_directory["hit_y"],batch_size=batch_size),
            iterate_array(entry_directory["hit_z"],batch_size=batch_size),
            iterate_array(entry_directory["hit_e"],batch_size=batch_size)
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

    return {
        "mean":MEAN,
        "histogram":hists,
        "variance" : VARIANCE,
        "std":STD,
        "max":MAX,
        "min":MIN,
        "total_points":TOTAL_PARTICLES
    }

def plot_analysis(analysis_dict:dict,GeV:int,root_dir):

    fig = plt.figure(constrained_layout = True,dpi=100)
    fig.set_size_inches(20,40)

    grid_spec = fig.add_gridspec(4,2)

    fig.suptitle("{} GeV Particle Stats".format(GeV),fontsize=72)

    for ind,feature in enumerate(FEATURES):

        plot_axes = fig.add_subplot(grid_spec[ind,0])
        stats_axes = fig.add_subplot(grid_spec[ind,1])

        hist = analysis_dict["histogram"][ind]
        plot_axes.bar(
            np.linspace(
                analysis_dict["min"][ind],
                analysis_dict["max"][ind],
                len(hist)
            ).astype(np.int32),
            hist,
            width= 10
        )
        plot_axes.set_title(feature, fontsize=64)
        plot_axes.set_xlabel(UNIT_OF_MEASURES[ind],fontsize=40)
        plot_axes.set_ylabel("# Particles",fontsize=40)

        stats_axes.set_title("{} Stats".format(feature), fontsize=48)
        stats_axes.grid(False)
        stats_axes.axes.xaxis.set_ticks([])
        stats_axes.axes.yaxis.set_ticks([])
        stats_axes.text(0.1, 0.5, show_stats(analysis_dict,ind), clip_on=True, fontsize=32)

    plt.savefig(os.path.join(root_dir,"{}GeV_particle_stats.png".format(GeV)))
    plt.close(fig)


def main():

    PLOTS_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..","..","plots"
    )


    for GeV in reversed(DATASETS.keys()):


        for dataset in DATASETS[GeV]:

            entry_strat = dataset.get("entries",None) or entries
            dataset_path = dataset["path"]


            with uproot.open(dataset_path) as root:

                entry_directory = entry_strat(root)
                analysis_dict = analyze_dataset(entry_directory)
                plot_analysis(analysis_dict,GeV,PLOTS_PATH)