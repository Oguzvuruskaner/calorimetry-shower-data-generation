import uproot
import numpy as np
from tqdm import tqdm
from math import sqrt
from os import path

__ROOT_DIRECTORY__ =  b"showers"

MAX_COLLISION_IN_EXPERIMENT = 200000



def create_all_inputs_file(pathList:[str]):
    """
    All features are combined into their feature array
    regardless of experiment block.

    :param pathList: List of root file paths.
    :return: None
    """

    hit_e = np.array([])
    hit_r = np.array([])
    hit_z = np.array([])

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            tmp_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())
            tmp_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            tmp_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            tmp_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())

        currentElement = 0
        for i in range(len(tmp_e)):
            currentElement += len(tmp_e[i])


        tmp = np.empty((currentElement, 1))
        currentElement = 0
        for a in tqdm(tmp_e):
            for b in a:
                tmp[currentElement][0] = b
                currentElement += 1
        hit_e = np.append(tmp,hit_e)


        tmp = np.empty((currentElement, 1))
        currentElement = 0
        for x_array,y_array in tqdm(zip(tmp_x,tmp_y)):
            for x_val,y_val in zip(x_array,y_array):
                tmp[currentElement][0] = sqrt(x_val*x_val + y_val*y_val)
                currentElement += 1

        hit_r = np.append(tmp, hit_r)

        tmp = np.empty((currentElement, 1))
        currentElement = 0
        for a in tqdm(tmp_z):
            for b in a:
                tmp[currentElement][0] = b
                currentElement += 1

        hit_z = np.append(tmp, hit_z)


    hit_e.resize((hit_e.size,1))
    hit_r.resize((hit_r.size,1))
    hit_z.resize((hit_z.size,1))

    np.save(path.join("npy","hit_e_combined.npy"), hit_e)
    np.save(path.join("npy","hit_r_combined.npy"), hit_r)
    np.save(path.join("npy","hit_z_combined.npy"), hit_z)


def create_per_experiment_file(pathList:[str]):

    """
    Splits hit_x, hit_y, hit_z, hit_e.

    :param pathList:List of root file paths.
    :return: None
    """

    hit_e = np.array([])
    hit_z = np.array([])
    hit_r = np.array([])

    for rootFile in pathList:

        with uproot.open(rootFile) as root:
            hit_e = np.append(hit_e, np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array()))
            hit_z = np.append(hit_z, np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array()))

            tmp_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            tmp_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())

            tmp = []
            for x_array,y_array in tqdm(zip(tmp_x,tmp_y)):
                experiment_tmp = []

                for x_val,y_val in zip(x_array,y_array):
                    experiment_tmp.append(sqrt(x_val*x_val + y_val*y_val))

                tmp.append(experiment_tmp)

            hit_r = np.append(hit_r,tmp)

    np.save(path.join("npy","hit_e.npy"), hit_e)
    np.save(path.join("npy","hit_r.npy"), hit_r)
    np.save(path.join("npy","hit_z.npy"), hit_z)


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
    np.save(path.join("npy","triple_all.npy"), quadruple_array)



