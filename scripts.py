from typing import Union
from scipy.stats import describe
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
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


__DATASETS__ = ["quadruple_all","hit_x_combined","hit_y_combined","hit_z_combined","hit_e_combined"]

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



def createScalers():

    for dataset_name in tqdm(__DATASETS__):

        data = np.load(path.join("npy","{}.npy".format(dataset_name)))
        train_preprocessors(data,dataset_name)


def plotFeatures(NUMBER_OF_BINS=100,plot=False):

    __SCALERS__ = ["min_max_scaler","robust_scaler","standard_scaler","max_abs_scaler"]

    for dataset_name in __DATASETS__:
        data :np.array = np.load(path.join("npy","{}.npy".format(dataset_name)),allow_pickle=True)

        if data.shape[1] == 1 :

            for scaler_name in __SCALERS__:

                with open(path.join("scalers","{}_{}.pkl".format(dataset_name,scaler_name)),"rb") as fp:
                    scaler : Union[MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler] = load(fp)

                plt.hist(scaler.transform(data),NUMBER_OF_BINS)
                plt.xlabel("Value")
                plt.ylabel("# Occurences")
                plt.title("{} {}".format(dataset_name,scaler_name))
                if not plot:
                    plt.savefig(path.join("plots", "{}_{}.png".format(dataset_name, scaler_name)), bbox_inches='tight')
                else:
                    plt.plot(bbox_inches='tight')
