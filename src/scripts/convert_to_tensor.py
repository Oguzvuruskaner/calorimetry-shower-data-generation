import numpy as np
import torch

import os

from tqdm import tqdm

from src.config import *
from multiprocessing import Pool

def proc_read(param ):

    file_ind,file = param

    return file_ind,np.load(file,allow_pickle=True)[0]



def convert_to_tensor():

    MATRIX_DATA_ROOT = os.path.join("..","data","matrix_dataset")

    with Pool(8) as pool:

        files = [
            os.path.join(MATRIX_DATA_ROOT,basename)
            for basename in os.listdir(MATRIX_DATA_ROOT)
            if basename.endswith(".npy")
        ]


        arrr = np.zeros((len(files),MATRIX_DIMENSION,MATRIX_DIMENSION),dtype=np.float32)

        for file_ind,tmp_array in pool.imap(proc_read,enumerate(tqdm(files))):
            #First index includes a ndarray which is jet image.
            #Second index includes all particles of jet.
            arrr[file_ind,...] = tmp_array
            del tmp_array




        tensor = torch.from_numpy(arrr)
        torch.save(tensor,os.path.join("..","data","jet_images.pt"))
