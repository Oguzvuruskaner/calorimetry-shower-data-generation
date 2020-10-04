from random import choice

import uproot4 as uproot
from tqdm import  tqdm

from src.datasets import DATASETS, entries
import numpy as np
import tables

import os
from src.config import *


from src.utils import iterate_array

DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data")
MATRIX_DATASET_FOLDER = os.path.join(DATA_FOLDER, "matrix_dataset")
TENSOR_DATASET_FOLDER = os.path.join(DATA_FOLDER, "tensor_dataset")


def open_h5_files() -> (list,list):


    float_atom = tables.Float32Atom()
    int_atom = tables.Int32Atom()

    fd_m = tables.open_file(os.path.join(MATRIX_DATASET_FOLDER, "all.h5"), mode="w")
    data_m =fd_m.create_earray(fd_m.root, "data", float_atom, (0, MATRIX_DIMENSION, MATRIX_DIMENSION),
                       expectedrows=600000)
    label_m = fd_m.create_earray(fd_m.root, "labels", int_atom, (0, 1), expectedrows=600000)

    fd_t = tables.open_file(os.path.join(TENSOR_DATASET_FOLDER, "all.h5"), mode="w")

    data_t = fd_t.create_earray(fd_t.root,"data",float_atom,(0,TENSOR_DIMENSION,TENSOR_DIMENSION,
                                                             TENSOR_DIMENSION),expectedrows=300000)
    label_t = fd_t.create_earray(fd_t.root,"labels",int_atom,(0,1),expectedrows=300000)


    return (fd_m,data_m,label_m),(fd_t,data_t,label_t)

def create_h5_files(batch_size = 1000):



    (fd_m,data_m,label_m),(fd_t,data_t,label_t) = open_h5_files()



    for GeV in DATASETS.keys():
        gev_array = np.array([GeV]).reshape(1,1)

        for dataset in DATASETS[GeV]:

            entry_strat = dataset.get("entries",None) or entries
            dataset_path = dataset["path"]

            with uproot.open(dataset_path) as root:


                entry_directory = entry_strat(root)

                for ind, (x_batch, y_batch, z_batch, e_batch) in tqdm(enumerate(zip(
                        iterate_array(entry_directory["hit_x"], batch_size=batch_size),
                        iterate_array(entry_directory["hit_y"], batch_size=batch_size),
                        iterate_array(entry_directory["hit_z"], batch_size=batch_size),
                        iterate_array(entry_directory["hit_e"], batch_size=batch_size)
                ))):

                    for hit_x,hit_y,hit_z,hit_e in zip(x_batch,y_batch,z_batch,e_batch):

                        hit_x = (hit_x-HIT_X_MIN) / (HIT_X_MAX-HIT_X_MIN)
                        hit_y = (hit_y-HIT_Y_MIN) / (HIT_Y_MAX-HIT_Y_MIN)
                        hit_z = (hit_z-HIT_Z_MIN) / (HIT_Z_MAX-HIT_Z_MIN)
                        hit_e /= GeV
                        hit_r = (hit_x*hit_x + hit_y*hit_y)**.5

                        matrix_view = np.hstack([
                            hit_r.reshape(-1,1),
                            hit_z.reshape(-1,1),
                            hit_e.reshape(-1,1)
                        ])

                        tensor_view = np.hstack([
                            hit_x.reshape(-1, 1),
                            hit_y.reshape(-1, 1),
                            hit_z.reshape(-1, 1),
                            hit_e.reshape(-1, 1)
                        ])

                        hist_2d = np.histogramdd(matrix_view[:, :2],
                                                 bins=MATRIX_DIMENSION,
                                                 range=np.array([[0, 1], [0, 1]]),
                                                 weights=matrix_view[:, 2])[0]

                        data_m.append(hist_2d.reshape((1,MATRIX_DIMENSION,MATRIX_DIMENSION)))
                        label_m.append(gev_array)

                        hist_3d = np.histogramdd(tensor_view[:, :3],
                                                 bins=TENSOR_DIMENSION,
                                                 range=np.array([[0, 1], [0, 1], [0, 1]]),
                                                 weights=tensor_view[:, 3])[0]

                        data_t.append(hist_3d.reshape((1,TENSOR_DIMENSION,TENSOR_DIMENSION,TENSOR_DIMENSION)))
                        label_t.append(gev_array)




    fd_m.close()
    fd_t.close()

