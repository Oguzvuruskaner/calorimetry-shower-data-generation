import uproot4 as uproot
from tqdm import  tqdm

from src.datasets import DATASETS, entries
import numpy as np
import tables

import os
from src.config import *


from src.utils import iterate_array, bernoulli

DATA_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "data")
PARTICLE_DATASET_FOLDER = os.path.join(DATA_FOLDER, "particle_dataset")



def create_particle_dataset(batch_size = 1000):

    float_atom = tables.Float32Atom()
    int_atom = tables.Int32Atom()

    fd = tables.open_file(os.path.join(PARTICLE_DATASET_FOLDER, "all.h5"), mode="w")
    data = fd.create_vlarray(fd.root, "data", float_atom,  expectedrows=600000)
    labels = fd.create_earray(fd.root, "labels", int_atom, (0, 1), expectedrows=600000)

    fd_test = tables.open_file(os.path.join(PARTICLE_DATASET_FOLDER, "all_test.h5"), mode="w")
    data_test = fd_test.create_vlarray(fd_test.root, "data", float_atom,
                                     expectedrows=600000)
    label_test = fd_test.create_earray(fd_test.root, "labels", int_atom, (0, 1), expectedrows=600000)


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


                        hit_x = hit_x[hit_e != 0]
                        hit_y = hit_y[hit_e != 0]
                        hit_z = hit_z[hit_e != 0]
                        hit_e = hit_e[hit_e != 0]

                        hit_x = (hit_x-HIT_X_MIN) / (HIT_X_MAX-HIT_X_MIN) * 2 -1
                        hit_y = (hit_y-HIT_Y_MIN) / (HIT_Y_MAX-HIT_Y_MIN) * 2 -1
                        hit_z = (hit_z-HIT_Z_MIN) / (HIT_Z_MAX-HIT_Z_MIN) * 2 -1
                        hit_e /= GeV

                        particles = np.stack([hit_x,hit_y,hit_z,hit_e],axis=1)
                        #Sorting of particles.
                        particles = particles[np.argsort(np.sum(particles[,:3]**2,axis=1))]



                        if bernoulli(0.9):
                            data.append(particles.reshape((-1,)))
                            labels.append(gev_array)
                        else:
                            data_test.append(particles.reshape((-1,)))
                            label_test.append(gev_array)




    fd.close()
    fd_test.close()
