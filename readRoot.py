import uproot
import numpy as np
from tqdm import tqdm
import os
from config import HIT_R_MAX,HIT_R_MIN,HIT_Z_MAX,HIT_Z_MIN,DIMENSION
from PIL import Image
import math

__ROOT_DIRECTORY__ =  b"showers"

from scripts.test_model import plot_data

MAX_COLLISION_IN_EXPERIMENT = 200000



def create_all_hits_file(pathList:[str]):
    """

    :param pathList: List of root file paths.
    :return: None
    """
    total_element = 0

    for rootFile in pathList:

        with uproot.open(rootFile) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())

            for i in range(len(hit_x)):
                total_element += len(hit_x[i])


    all_hits = np.zeros((total_element,3))

    current_element = 0

    for rootFile in pathList:

        with uproot.open(rootFile) as root:

            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())



            for ind in tqdm(range(len(hit_x))):

                all_hits[current_element:current_element+len(hit_x[ind]),0] = np.sqrt(hit_x[ind]*hit_x[ind] + hit_y[ind]*hit_y[ind])
                all_hits[current_element:current_element+len(hit_x[ind]),1] = hit_z[ind]
                all_hits[current_element:current_element+len(hit_x[ind]),2] = hit_e[ind]
                current_element += len(hit_x[ind])

    np.save(os.path.join("npy","triple_all.npy"), all_hits,allow_pickle=True)
    print("Saving all hits file finished.")

def create_per_jet_file(root_files:[str]):

    jet_list = []

    for root_file in root_files:

        with uproot.open(root_file) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())


        for i in tqdm(range(len(hit_x))):
            tmp_jet = np.zeros((len(hit_x[i]),3))

            tmp_jet[:,0] = np.sqrt(hit_x[i]* hit_x[i] + hit_y[i] * hit_y[i])
            tmp_jet[:,1] = hit_z[i]
            tmp_jet[:,2] = hit_e[i]

            jet_list.append(tmp_jet)

    np.save(os.path.join("npy","per_jet_all.npy"),np.array(jet_list))


def create_jet_plots(root_files:[str]):

    counter = 1

    for root_file in root_files:

        with uproot.open(root_file) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())


        for i in tqdm(range(len(hit_x))):
            tmp_jet = np.zeros((len(hit_x[i]),3))

            tmp_jet[:,0] = np.sqrt(hit_x[i]* hit_x[i] + hit_y[i] * hit_y[i])
            tmp_jet[:,1] = hit_z[i]
            tmp_jet[:,2] = hit_e[i]

            plot_data(tmp_jet,"Jet {}".format(counter), os.path.join("jet_images","plots" ,"{}.png".format(counter)),jet_bins=100, jet=True,dpi=500)
            counter += 1



def create_jet_images(root_files: [str]):

    counter = 1

    for root_file in root_files:

        with uproot.open(root_file) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())


        for i in tqdm(range(len(hit_x))):

            tmp_jet = np.zeros((len(hit_x[i]), 3))
            image = np.zeros((DIMENSION,DIMENSION))

            tmp_jet[:, 0] = np.sqrt(hit_x[i] * hit_x[i] + hit_y[i] * hit_y[i])
            tmp_jet[:, 1] = hit_z[i]
            tmp_jet[:, 2] = hit_e[i]

            #I didn't use classical normalization
            #Normalize globally.
            tmp_jet[:, 0] = np.floor((tmp_jet[:, 0] - HIT_R_MIN) / (HIT_R_MAX - HIT_R_MIN) * DIMENSION)
            tmp_jet[:, 1] = np.floor((tmp_jet[:, 1] - HIT_Z_MIN) / (HIT_Z_MAX - HIT_Z_MIN) * DIMENSION)

            for r,z,e in tmp_jet:
                image[int(z),int(r)] += e


            # Image is divided by since it is
            # 50 GeV
            for i in range(len(image)):
                for j in range(len(image[i])):
                    image[i][j] = 0 if image[i][j] == 0 else (math.log10(image[i][j])+8)*32

            image = np.array(image,dtype=np.uint8)
            image = 255 - image

            img = Image.fromarray(image,"L")
            img.save(os.path.join("jet_images","images", "{}.png".format(counter)))

            counter += 1


def create_jet_image_array(root_files : [str]):

    total_jets = 0

    for rootFile in root_files:

        with uproot.open(rootFile) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())

            total_jets += len(hit_x)

    all_jets = np.zeros((total_jets, DIMENSION,DIMENSION))
    jet_counter = 0


    for root_file in root_files:

        with uproot.open(root_file) as root:
            hit_x = np.array(root[__ROOT_DIRECTORY__][b"hit_x"].array())
            hit_y = np.array(root[__ROOT_DIRECTORY__][b"hit_y"].array())
            hit_z = np.array(root[__ROOT_DIRECTORY__][b"hit_z"].array())
            hit_e = np.array(root[__ROOT_DIRECTORY__][b"hit_e"].array())

        for i in tqdm(range(len(hit_x))):

            tmp_jet = np.zeros((len(hit_x[i]), 3))

            tmp_jet[:, 0] = np.sqrt(hit_x[i] * hit_x[i] + hit_y[i] * hit_y[i])
            tmp_jet[:, 1] = hit_z[i]
            tmp_jet[:, 2] = hit_e[i]

            # I didn't use classical normalization
            # Normalize globally.
            tmp_jet[:, 0] = np.floor((tmp_jet[:, 0] - HIT_R_MIN) / (HIT_R_MAX - HIT_R_MIN) * DIMENSION)
            tmp_jet[:, 1] = np.floor((tmp_jet[:, 1] - HIT_Z_MIN) / (HIT_Z_MAX - HIT_Z_MIN) * DIMENSION)



            # Normalizing energy values.
            # It is known that total energy of jets is 50 GeV.
            all_jets[jet_counter] = np.histogram2d(
                x = tmp_jet[:, 0],
                y = tmp_jet[:, 1],
                weights = tmp_jet[:,2],
                bins = DIMENSION
            )[0] / 50
            jet_counter += 1


    np.save(
        os.path.join("npy","all_jet_images.npy"),
        all_jets,
        allow_pickle=True)