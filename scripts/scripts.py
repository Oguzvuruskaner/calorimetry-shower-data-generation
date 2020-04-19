import numpy as np
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from os import path,listdir,getcwd
from readRoot import create_all_inputs_file,\
    create_all_quadruple_file,\
    create_per_experiment_file
from config import __DATASETS_WITH_OUTLIERS__ ,__DATASETS__
import seaborn as sns

# scripts file include project specific
# functionalities.



def createNpyFiles():

    #Get all root files in root_files folder.
    root_files_directory = path.join(getcwd(),"root_files")
    root_files = [path.join("../root_files", root_file) for root_file in listdir(root_files_directory)
                  if root_file.endswith(".root")]

    create_per_experiment_file(root_files)
    create_all_quadruple_file(root_files)
    create_all_inputs_file(root_files)

def loadAndSplitArray(filepath:str,number_of_chunks):

    filenamePattern = re.compile(r"\\?\/?(.*?)\.npy")

    data = np.load(filepath)
    chunks = np.array_split(data,number_of_chunks)

    #Gets npy/filename out of full path.
    rootFilepath = re.findall(filenamePattern,filepath)[0]
    rootFilepath = path.basename(rootFilepath)

    for index,chunk  in enumerate(tqdm(chunks)):

        np.save("train_chunks/{}_chunk_{}.npy".format(rootFilepath,index+1),chunk)


def filterOutliers(outlier_threshold=4):
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


def plotFeatures():


    for dataset_name in __DATASETS__:
        data :np.array = np.load(path.join("../npy", "{}.npy".format(dataset_name)), allow_pickle=True)

        if data.shape[1] == 1 :
            fig = sns.distplot(data,kde=False)
            fig.savefig(path.join("../plots", "{}.png".format(dataset_name)), bbox_inches='tight')

