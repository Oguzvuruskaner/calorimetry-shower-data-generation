import uproot
import numpy as np
from tqdm import tqdm
from math import sqrt
import os

__ROOT_DIRECTORY__ =  b"showers"

MAX_COLLISION_IN_EXPERIMENT = 200000


def create_all_quadruple_file(pathList:[str]):
    """

    :param pathList: List of root file paths.
    :return: None
    """


    quadruple_array = np.array([],dtype=np.float64)

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())


        currentElement = 0
        for i in range(len(hit_x)):
            currentElement += len(hit_x[i])

        tmp = np.zeros((currentElement,3),dtype=np.float64)
        currentElement = 0

        for i in tqdm(range(len(hit_x))):
            for x_exp,y_exp,z_exp, e_exp in zip(hit_x[i],hit_y[i], hit_z[i], hit_e[i]):
                tmp[currentElement][0] = sqrt(x_exp*x_exp + y_exp*y_exp)
                tmp[currentElement][1] = z_exp
                tmp[currentElement][2] = e_exp

                currentElement += 1


        quadruple_array = np.append(quadruple_array, tmp)

    quadruple_array.resize((quadruple_array.size//3,3))
    np.save(os.path.join("npy","triple_all.npy"), quadruple_array)


def create_per_jet_file(root_files:[str]):

    jet_list = []

    for root_file in root_files:

        with uproot.open(root_file) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())


        for i in range(len(hit_x)):
            tmp_jet = np.zeros((len(hit_x[i]),3))

            for j in range(len(hit_x[i])):
                tmp_jet[j][0] = sqrt(hit_x[i][j] * hit_x[i][j] + hit_y[i][j] * hit_y[i][j])
                tmp_jet[j][1] = hit_z[i][j]
                tmp_jet[j][2] = hit_e[i][j]

            jet_list.append(tmp_jet)



    np.save(os.path.join("npy","per_jet_all.npy"),np.array(jet_list))




