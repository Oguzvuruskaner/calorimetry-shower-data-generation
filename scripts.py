from typing import Union
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from os import path,listdir,getcwd
from preprocessors import train_preprocessors
from pickle import load
from readRoot import create_all_inputs_file,\
    create_quadruple_array_file_fill_zeros,\
    create_all_quadruple_file,\
    create_per_experiment_file

# scripts file include project specific
# functionalities.


__DATASETS_WITH_OUTLIERS__ = ["quadruple_all","hit_x_combined","hit_y_combined","hit_z_combined","hit_e_combined"]

__DATASETS__ = ["quadruple_all","hit_x_combined","hit_y_combined","hit_z_combined","hit_e_combined",
                "quadruple_all_without_outliers","hit_x_combined_without_outliers",
                "hit_y_combined_without_outliers","hit_z_combined_without_outliers",
                "hit_e_combined_without_outliers"
                ]

# __DATASETS__ = ["hit_e_combined_without_outliers"]

def createNpyFiles():

    #Get all root files in root_files folder.
    root_files_directory = path.join(getcwd(),"root_files")
    root_files = [path.join("root_files",root_file) for root_file in listdir(root_files_directory)
                  if root_file.endswith(".root")]

    create_per_experiment_file(root_files)
    create_all_quadruple_file(root_files)
    create_all_inputs_file(root_files)
    create_quadruple_array_file_fill_zeros(root_files)

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
        data = np.load(path.join("npy", "{}.npy".format(dataset_name)), allow_pickle=True)
        ss = StandardScaler()
        ss.fit(data)
        filter_array = (ss.transform(data) <= outlier_threshold) | (ss.transform(data) >= -outlier_threshold)
        data = data[filter_array]
        data.resize((data.size,1))
        np.save(path.join("npy", "{}_without_outliers.npy".format(dataset_name)),data)


def createScalers():

    for dataset_name in tqdm(__DATASETS__):

        data = np.load(path.join("npy","{}.npy".format(dataset_name)))
        train_preprocessors(data,dataset_name)

def plotFeatures(NUMBER_OF_BINS=200,plot=False):

    __SCALERS__ = ["min_max_scaler","robust_scaler","standard_scaler","max_abs_scaler"]

    for dataset_name in __DATASETS__:
        data :np.array = np.load(path.join("npy","{}.npy".format(dataset_name)),allow_pickle=True)

        if data.shape[1] == 1 :

            plt.hist(data, NUMBER_OF_BINS)
            plt.xlabel("Value")
            plt.ylabel("# Occurences")
            plt.title("{}".format(dataset_name))
            if not plot:
                plt.savefig(path.join("plots", "{}.png".format(dataset_name)), bbox_inches='tight')
            else:
                plt.plot(bbox_inches='tight')

            for scaler_name in __SCALERS__:

                with open(path.join("scalers","{}_{}.pkl".format(dataset_name,scaler_name)),"rb") as fp:
                    scaler : Union[MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler] = load(fp)

                transformed_data = scaler.transform(data)

                plt.hist(transformed_data,NUMBER_OF_BINS)
                plt.xlabel("Value")
                plt.ylabel("# Occurences")
                plt.title("{} {}".format(dataset_name,scaler_name))
                if not plot:
                    plt.savefig(path.join("plots", "{}_{}.png".format(dataset_name, scaler_name)), bbox_inches='tight')
                else:
                    plt.plot(bbox_inches='tight')



def getSamples(sample_size=10**6):

    for dataset_name in tqdm(__DATASETS__):

        data = np.load(path.join("npy","{}.npy".format(dataset_name)),allow_pickle=True)

        if len(data.shape) <= 2:
            increment = data.shape[0]//sample_size
            if increment == 0:
                raise Exception("Size of sample : {} is bigger than dataset size {}.".format(sample_size,data.shape[0]))

            else:
                #Using systematic sampling to keep array distribution still.
                data = np.sort(data)
                np.save(path.join("npy","{}_{}.npy".format(dataset_name,sample_size)),data[::increment])

