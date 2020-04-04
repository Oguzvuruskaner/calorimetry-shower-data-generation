import numpy as np
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


def createNpyFiles():

    #Get all root files in root_files folder.
    root_files_directory = path.join(getcwd(),"root_files")
    root_files = [path.join("root_files",root_file) for root_file in listdir(root_files_directory) if root_file.endswith(".root")]

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

    data = np.load("npy/quadruple_all.npy", allow_pickle=True)
    train_preprocessors(data, "quadruple_all")
    print("1/5")

    data = np.load("npy/hit_x_combined.npy", allow_pickle=True)
    train_preprocessors(data, "hit_x_combined")
    print("2/5")

    data = np.load("npy/hit_y_combined.npy", allow_pickle=True)
    train_preprocessors(data, "hit_y_combined")
    print("3/5")

    data = np.load("npy/hit_z_combined.npy", allow_pickle=True)
    train_preprocessors(data, "hit_z_combined")
    print("4/5")

    data = np.load("npy/hit_e_combined.npy", allow_pickle=True)
    train_preprocessors(data, "hit_e_combined")
    print("5/5")


