from readRoot import create_npy_files
from Model import train, validate, validate_critic
import numpy as np
from os.path import join
from time import time

# Model Version
__VERSION__ = 3

if __name__ == "__main__":

    # Fully randomize the plotting process.
    np.random.seed(int(time()))

    create_npy_files("pion50GeVshowers.root")

    # allow_pickle parameter
    # prevents ValueError
    # resulting from numpy arrays.
    # arr = np.load("hit_e.npy",allow_pickle=True)

    # train(arr,version=__VERSION__)
    #
    # validate(arr,join("models","gen{}_generator.h5".format(__VERSION__)))

    # validate_critic(join("models","gen3_critic.h5"),join("models","gen3_generator.h5"))
