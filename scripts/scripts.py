import os


import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from readRoot import create_per_jet_file, create_jet_image_array, create_jet_images, \
    create_jet_plots
from config import __DATASETS_WITH_OUTLIERS__, HIT_R_MAX, HIT_R_MIN, HIT_Z_MAX, HIT_Z_MIN
from scripts.test_model import show_stats

from config import DIMENSION


# scripts file include project specific
# functionalities.

def scale_jets(jets:np.ndarray) -> None:

    for jet in jets:
        jet[:,0] = (jet[:,0] - HIT_R_MIN) / (HIT_R_MAX - HIT_R_MIN) * DIMENSION
        jet[:,1] = (jet[:,1]  - HIT_Z_MIN) / (HIT_Z_MAX - HIT_Z_MIN) * DIMENSION



def get_root_files():
    return [
        os.path.join("root_files", root_file)
        for root_file in os.listdir("root_files")
        if root_file.endswith(".root")
    ]

def create_npy_files():

    #Get all root files in root_files folder.
    root_files = [
        os.path.join("root_files", root_file)
        for root_file in os.listdir("root_files")
        if root_file.endswith(".root")
    ]

    create_per_jet_file(root_files)
    create_jet_images(root_files)
    create_jet_image_array(root_files)
    create_jet_plots(root_files)


def filter_outliers(outlier_threshold=4):
    """
    :param outlier_threshold: inside nth standard deviation
        values are normal values ,outside the nth standard deviation
        are abnormal values. .

        4th standard deviation includes 99.99% of Gaussian distribution.

    :return: None
    """

    for dataset_name in tqdm(__DATASETS_WITH_OUTLIERS__):
        data = np.load(os.path.join("../npy", "{}.npy".format(dataset_name)), allow_pickle=True)
        ss = StandardScaler()
        ss.fit(data)
        filter_array = (ss.transform(data) <= outlier_threshold) | (ss.transform(data) >= -outlier_threshold)
        data = data[filter_array]
        data.resize((data.size,1))
        np.save(os.path.join("../npy", "{}_without_outliers.npy".format(dataset_name)), data)



def create_jet_particles_plot(data):
    import seaborn as sns
    import matplotlib.pyplot as plt

    shapes = np.zeros((data.shape[0], 1))

    for i in range(len(data)):
        shapes[i] = data[i].shape[0]

    fig,[ax0,ax1] = plt.subplots(1,2)
    fig.set_size_inches(20, 10)

    sns.distplot(shapes,ax=ax0)
    ax1.text(0.1, 0.5, show_stats(shapes), clip_on=True, fontsize=24)

    plt.savefig(os.path.join("plots","jet_shapes"),dpi=500)
    plt.clf()