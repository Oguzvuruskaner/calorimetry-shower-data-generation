import uproot
import numpy as np

# __ROOT_DIRECTORIES__ = [b"showers;14", b"showers;15", b"showers"]
__ROOT_DIRECTORY__ =  b"showers"


def create_npy_files(pathList:str):

    hit_e = np.array([])
    hit_x = np.array([])
    hit_y = np.array([])
    hit_z = np.array([])

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            hit_e = np.append(hit_e,np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array()))
            hit_x = np.append(hit_x,np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array()))
            hit_y = np.append(hit_y,np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array()))
            hit_z = np.append(hit_z,np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array()))


    np.save("hit_e.npy",hit_e)
    np.save("hit_x.npy",hit_x)
    np.save("hit_y.npy",hit_y)
    np.save("hit_z.npy",hit_z)



def create_quadruple_array_file(pathList:list):

    """
    :param :path
    :return: None

    Merges 4 properties with given order
    (hit_x,hit_y,hit_z,hit_e)
    and writes them to a file.

    """

    quadruple_array = np.array([])

    for rootFile in pathList:

        tmp = []

        with uproot.open(rootFile) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())

            # Memory allocation of np.array increases the execution time.
            # Hence it is better to use linked list.
            # Linked list doesn't have to be reallocated in any
            # append operation.

            for x_exp,y_exp,z_exp,e_exp in zip(hit_x,hit_y,hit_z,hit_e):
                tmp.append([x_exp,y_exp,z_exp,e_exp])

            tmp = np.array(tmp)

        quadruple_array = np.append(quadruple_array,tmp)


    np.save("quadruple.npy",quadruple_array)