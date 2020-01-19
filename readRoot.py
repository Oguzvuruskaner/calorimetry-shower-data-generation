import uproot
import numpy as np




def create_npy_files(path:str):

    root = uproot.open(path)

    hit_e = root[b"showers;14"][b"hit_e"].array()._content
    hit_x = root[b"showers;14"][b"hit_x"].array()._content
    hit_y = root[b"showers;14"][b"hit_y"].array()._content
    hit_z = root[b"showers;14"][b"hit_z"].array()._content

    np.append(hit_e,root[b"showers;15"][b"hit_e"].array()._content)
    np.append(hit_x,root[b"showers;15"][b"hit_x"].array()._content)
    np.append(hit_y,root[b"showers;15"][b"hit_y"].array()._content)
    np.append(hit_z,root[b"showers;15"][b"hit_z"].array()._content)

    np.save("hit_e.npy",hit_e)
    np.save("hit_x.npy",hit_x)
    np.save("hit_y.npy",hit_y)
    np.save("hit_z.npy",hit_z)

