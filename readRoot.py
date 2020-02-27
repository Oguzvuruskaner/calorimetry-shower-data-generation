import uproot
import numpy as np

# __ROOT_DIRECTORIES__ = [b"showers;14", b"showers;15", b"showers"]
__ROOT_DIRECTORIES__ = [ b"showers"]


def create_npy_files(path:str):

    root = uproot.open(path)


    hit_e = np.array([])
    hit_x = np.array([])
    hit_y = np.array([])
    hit_z = np.array([])

    for directory in __ROOT_DIRECTORIES__:

        hit_e = np.append(hit_e,np.array(root[directory][b"hit_e"].array()))
        hit_x = np.append(hit_x,np.array(root[directory][b"hit_x"].array()))
        hit_y = np.append(hit_y,np.array(root[directory][b"hit_y"].array()))
        hit_z = np.append(hit_z,np.array(root[directory][b"hit_z"].array()))


    np.save("hit_e.npy",hit_e)
    np.save("hit_x.npy",hit_x)
    np.save("hit_y.npy",hit_y)
    np.save("hit_z.npy",hit_z)



def create_quadruple_array_file(path:str):

    """
    :param :path
    :return: None

    Merges 4 properties with given order
    (hit_x,hit_y,hit_z,hit_e)
    and writes them to a file.

    """
    root = uproot.open(path)

    quadruple_array = np.array([])

    for directory in __ROOT_DIRECTORIES__:

        hit_x = np.array(root[directory][b"hit_x"].array())
        hit_y = np.array(root[directory][b"hit_y"].array())
        hit_z = np.array(root[directory][b"hit_z"].array())
        hit_e = np.array(root[directory][b"hit_e"].array())

        tmp = np.ndarray([])

        for x_exp,y_exp,z_exp,e_exp in zip(hit_x,hit_y,hit_z,hit_e):
            tmp = np.append(tmp,list(zip(x_exp,y_exp,z_exp,e_exp)))

        quadruple_array = np.append(quadruple_array,tmp)


    np.save("quadruple.npy",quadruple_array)