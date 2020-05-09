import uproot
import numpy as np
from tqdm import tqdm
from math import sqrt
from os import path

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
    np.save(path.join("npy","triple_all.npy"), quadruple_array)



