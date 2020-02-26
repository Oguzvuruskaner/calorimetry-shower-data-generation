import uproot
import numpy as np




def create_npy_files(path:str):

    root = uproot.open(path)

    rootDirectories = [b"showers;14",b"showers;15",b"showers"]

    hit_e = np.array([])
    hit_x = np.array([])
    hit_y = np.array([])
    hit_z = np.array([])

    for directory in rootDirectories:

        hit_e = np.append(hit_e,np.array(root[directory][b"hit_e"].array()))
        hit_x = np.append(hit_x,np.array(root[directory][b"hit_x"].array()))
        hit_y = np.append(hit_y,np.array(root[directory][b"hit_y"].array()))
        hit_z = np.append(hit_z,np.array(root[directory][b"hit_z"].array()))


    np.save("hit_e.npy",hit_e)
    np.save("hit_x.npy",hit_x)
    np.save("hit_y.npy",hit_y)
    np.save("hit_z.npy",hit_z)

