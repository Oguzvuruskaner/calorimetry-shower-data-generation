import numpy as np
from sklearn.preprocessing import  StandardScaler
from tqdm import tqdm
import re
from os import path,listdir,getcwd
from readRoot import create_all_quadruple_file
from config import __DATASETS_WITH_OUTLIERS__ ,__DATASETS__
import seaborn as sns
from scripts.test_model import plot_data

# scripts file include project specific
# functionalities.

def plot_all_data():

    data = np.load(path.join("npy","triple_all.npy"))
    plot_data(data[:,0],"All R Data",path.join("plots","all_r_data.png"))
    plot_data(data[:,1],"All Z Data",path.join("plots","all_z_data.png"))
    plot_data(data[:,2],"All E Data",path.join("plots","all_e_data.png"))



def create_npy_files():

    #Get all root files in root_files folder.
    root_files_directory = path.join(getcwd(),"root_files")
    root_files = [path.join("../root_files", root_file) for root_file in listdir(root_files_directory)
                  if root_file.endswith(".root")]

    create_all_quadruple_file(root_files)


def filter_outliers(outlier_threshold=4):
    """
    :param outlier_threshold: inside nth standard deviation
        values are normal values ,outside the nth standard deviation
        are abnormal values. .

        4th standard deviation includes 99.99% of Gaussian distribution.

    :return: None
    """

    for dataset_name in tqdm(__DATASETS_WITH_OUTLIERS__):
        data = np.load(path.join("../npy", "{}.npy".format(dataset_name)), allow_pickle=True)
        ss = StandardScaler()
        ss.fit(data)
        filter_array = (ss.transform(data) <= outlier_threshold) | (ss.transform(data) >= -outlier_threshold)
        data = data[filter_array]
        data.resize((data.size,1))
        np.save(path.join("../npy", "{}_without_outliers.npy".format(dataset_name)), data)


