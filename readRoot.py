import uproot
import numpy as np
from tqdm import tqdm

# __ROOT_DIRECTORIES__ = [b"showers;14", b"showers;15", b"showers"]
__ROOT_DIRECTORY__ =  b"showers"


def create_all_inputs_file(pathList:[str]):
    """
    All features are combined into their feature array
    regardless of experiment block.

    :param pathList: List of root file paths.
    :return: None
    """

    hit_e = np.array([])
    hit_x = np.array([])
    hit_y = np.array([])
    hit_z = np.array([])

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            hit_e = np.append(hit_e, np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array()).flatten())
            hit_x = np.append(hit_x, np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array()).flatten())
            hit_y = np.append(hit_y, np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array()).flatten())
            hit_z = np.append(hit_z, np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array()).flatten())


    np.save("hit_e_combined.npy", hit_e)
    hit_e = None
    np.save("hit_x_combined.npy", hit_x)
    hit_x = None
    np.save("hit_y_combined.npy", hit_y)
    hit_y = None
    np.save("hit_z_combined.npy", hit_z)
    hit_z = None


def create_per_experiment_file(pathList:[str]):

    """
    Splits hit_x, hit_y, hit_z, hit_e.

    :param pathList:List of root file paths.
    :return: None
    """

    hit_e = np.array([])
    hit_x = np.array([])
    hit_y = np.array([])
    hit_z = np.array([])

    for rootFile in pathList:

        with uproot.open(rootFile) as root:
            hit_e = np.append(hit_e, np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array()))
            hit_x = np.append(hit_x, np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array()))
            hit_y = np.append(hit_y, np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array()))
            hit_z = np.append(hit_z, np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array()))


    np.save("hit_e.npy", hit_e)
    hit_e = None
    np.save("hit_x.npy", hit_x)
    hit_x = None
    np.save("hit_y.npy", hit_y)
    hit_y = None
    np.save("hit_z.npy", hit_z)
    hit_z = None

def create_all_quadruple_array_file(pathList:[str]):
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

        tmp = np.empty((currentElement,4),dtype=np.float64)
        currentElement = 0

        for i in tqdm(range(len(hit_x))):
            for x_exp, y_exp, z_exp, e_exp in zip(hit_x[i], hit_y[i], hit_z[i], hit_e[i]):
                tmp[currentElement][0] = x_exp
                tmp[currentElement][1] = y_exp
                tmp[currentElement][2] = z_exp
                tmp[currentElement][3] = e_exp

                currentElement += 1


        quadruple_array = np.append(quadruple_array, tmp)
        tmp = []

    np.save("quadruple_all.npy", quadruple_array)


def create_quadruple_array_file(pathList:[str]):

    """
    Merges 4 features with given order
    (hit_x,hit_y,hit_z,hit_e)
    and writes them to a file.

    :param :List of root file paths.
    :return: None
    """

    quadruple_array = np.array([])

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())

        tmp2 = []

        for i in tqdm(range(len(hit_x))):
            tmp = []

            for x_exp,y_exp,z_exp,e_exp in zip(hit_x[i],hit_y[i],hit_z[i],hit_e[i]):
                tmp.append([x_exp,y_exp,z_exp,e_exp])

            tmp2.append(np.array(tmp))

        quadruple_array = np.append(quadruple_array,tmp2)


    np.save("quadruple.npy", quadruple_array)