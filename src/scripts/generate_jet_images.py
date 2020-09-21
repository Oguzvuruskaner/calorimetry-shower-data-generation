import numpy as np
import uproot as uproot
import os
from math import ceil

from tqdm import tqdm, trange

from src.config import *

NPY_DIR = os.path.join("..","npy")
ROOT_DIR = os.path.join("..","root_files")

ARRAY_NAMES = [
    "hit_x",
    "hit_y",
    "hit_z",
    "hit_e"
]



def generate():

    IMAGE_NUMBER = 1

    root_files = [
        os.path.join(ROOT_DIR,filename)
        for filename in os.listdir(ROOT_DIR)
        if filename.endswith(".root")
    ]


    for file_index,root_file in enumerate(root_files):



        with uproot.open(root_file) as root:

            NUMBER_OF_ENTRIES = root[b"showers"].numentries

        # with uproot.open(root_file) as root:
        #
        #     for name in tqdm(ARRAY_NAMES):
        #         np.save(os.path.join("..","npy","{}_{}.npy".format(name,file_index)), np.array(root[b"showers"]["{}".format(name).encode()].array()))
        #

        NUMBER_OF_ITERATIONS = ceil(NUMBER_OF_ENTRIES // DATAPOINT_PER_FILE)


        for iteration_index in trange(NUMBER_OF_ITERATIONS):

            tmp_array = []

            for name in ARRAY_NAMES:

                tmp = np.load(os.path.join("..","npy","{}_{}.npy".format(name,file_index)),allow_pickle=True)
                tmp_array.append( np.copy(tmp[iteration_index*DATAPOINT_PER_FILE:
                                     min(NUMBER_OF_ENTRIES,(iteration_index+1)*DATAPOINT_PER_FILE)] ))

                del tmp

            hit_x = tmp_array[0]
            hit_y = tmp_array[1]
            hit_z = tmp_array[2]
            hit_e = tmp_array[3]

            for ind in range(len(hit_x)):

                tmp_x = hit_x[ind]/HIT_X_MAX
                tmp_y = hit_y[ind]/HIT_Y_MAX
                tmp_r = np.sqrt(hit_x[ind] * hit_x[ind] + hit_y[ind] * hit_y[ind]) / HIT_R_MAX
                tmp_z = (hit_z[ind] -HIT_Z_MIN)/(HIT_Z_MAX-HIT_Z_MIN)
                tmp_e = hit_e[ind]/HIT_E_MAX

                matrix_view = np.stack([tmp_r,tmp_z,tmp_e],axis=1)
                tensor_view = np.stack([tmp_x,tmp_y,tmp_z,tmp_e],axis=1)

                hist_2d = np.histogramdd(matrix_view[:,:2],
                                         bins=MATRIX_DIMENSION,
                                         range=np.array([[0,1],[0,1]]),
                                         weights=matrix_view[:,2])[0]
                hist_3d = np.histogramdd(tensor_view[:,:3],
                                         bins=TENSOR_DIMENSION,
                                         range=np.array([[0,1],[0,1],[0,1]]),
                                         weights=tensor_view[:,3])[0]

                np.save(os.path.join(NPY_DIR,"matrix_dataset","{}.npy".format(IMAGE_NUMBER)),np.array([hist_2d,matrix_view]))
                np.save(os.path.join(NPY_DIR,"tensor_dataset","{}.npy".format(IMAGE_NUMBER)),np.array([hist_3d,tensor_view]))
                IMAGE_NUMBER +=1