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


def open_h5_files(number_of_matrix_files,number_of_tensor_files) -> (list,list):

    matrix_files = []
    tensor_files = []

    float_atom = tables.Float32Atom()
    int_atom = tables.Int32Atom()

    for ind in range(number_of_matrix_files):

        x_fd = tables.open_file(os.path.join(MATRIX_DATASET_FOLDER,"{}_x.h5".format(ind)),mode="w")
        y_fd = tables.open_file(os.path.join(MATRIX_DATASET_FOLDER,"{}_y.h5".format(ind)),mode="w")
        x_fd.create_earray(x_fd.root,"data_{}".format(ind),float_atom,(0,MATRIX_DIMENSION,MATRIX_DIMENSION),expectedrows=60000)
        y_fd.create_earray(y_fd.root,"labels_{}".format(ind),int_atom,(0,1),expectedrows=60000)

        matrix_files.append((x_fd,y_fd))

    for ind in range(number_of_tensor_files):

        x_fd = tables.open_file(os.path.join(TENSOR_DATASET_FOLDER,"{}_x.h5".format(ind)),mode="w")
        y_fd = tables.open_file(os.path.join(TENSOR_DATASET_FOLDER,"{}_y.h5".format(ind)),mode="w")
        x_array = x_fd.create_earray(x_fd.root,"data_{}".format(ind),float_atom,(0,TENSOR_DIMENSION,TENSOR_DIMENSION,TENSOR_DIMENSION),expectedrows=30000)
        y_array = y_fd.create_earray(y_fd.root,"labels_{}".format(ind),int_atom,(0,1),expectedrows=30000)

        tensor_files.append(((x_fd,x_array),(y_fd,y_array)))

    return matrix_files,tensor_files

def create_h5_files(number_of_matrix_files=10,number_of_tensor_files=20,batch_size = 1000):



    matrix_files,tensor_files = open_h5_files(number_of_matrix_files,number_of_tensor_files)



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

                        (_,x_array),(_,y_array) = choice(matrix_files)
                        x_array.append(hist_2d.reshape(1,MATRIX_DIMENSION,MATRIX_DIMENSION))
                        y_array.append(gev_array)

                        hist_3d = np.histogramdd(tensor_view[:, :3],
                                                 bins=TENSOR_DIMENSION,
                                                 range=np.array([[0, 1], [0, 1], [0, 1]]),
                                                 weights=tensor_view[:, 3])[0]

                        (_, x_array), (_, y_array) = choice(tensor_files)
                        x_array.append(hist_3d.reshape(1,TENSOR_DIMENSION,TENSOR_DIMENSION,TENSOR_DIMENSION))
                        y_array.append(gev_array)




    for ((x_fd,_),(y_fd,_)) in matrix_files:
        x_fd.close()
        y_fd.close()

    for ((x_fd,_),(y_fd,_)) in tensor_files:
        x_fd.close()
        y_fd.close()

