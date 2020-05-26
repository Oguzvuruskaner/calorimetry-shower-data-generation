import uproot
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from math import sqrt
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


__ROOT_DIRECTORY__ =  b"showers"

MAX_COLLISION_IN_EXPERIMENT = 200000


def create_all_hits_file(pathList:[str]):
    """

    :param pathList: List of root file paths.
    :return: None
    """
    total_element = 0

    for rootFile in pathList:

        with uproot.open(rootFile) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())

            for i in range(len(hit_x)):
                total_element += len(hit_x[i])


    all_hits = np.zeros((total_element,3))

    current_element = 0

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())



            for ind in tqdm(range(len(hit_x))):

                all_hits[current_element:current_element+len(hit_x[ind]),0] = np.sqrt(hit_x[ind]*hit_x[ind] + hit_y[ind]*hit_y[ind])
                all_hits[current_element:current_element+len(hit_x[ind]),1] = hit_z[ind]
                all_hits[current_element:current_element+len(hit_x[ind]),2] = hit_e[ind]
                current_element += len(hit_x[ind])

    np.save(os.path.join("npy","triple_all.npy"), all_hits,allow_pickle=True)
    print("Saving all hits file finished.")

def create_per_jet_file(root_files:[str]):

    jet_list = []

    for root_file in root_files:

        with uproot.open(root_file) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())


        for i in tqdm(range(len(hit_x))):
            tmp_jet = np.zeros((len(hit_x[i]),3))

            tmp_jet[:,0] = np.sqrt(hit_x[i]* hit_x[i] + hit_y[i] * hit_y[i])
            tmp_jet[:,1] = hit_z[i]
            tmp_jet[:,2] = hit_e[i]

            jet_list.append(tmp_jet)

    np.save(os.path.join("npy","per_jet_all.npy"),np.array(jet_list))

