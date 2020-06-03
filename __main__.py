from JetGenerator import train_model
from readRoot import create_jet_images

from scripts.scripts import get_root_files
from scripts.test_model import generate_jet_images, save_jet_image
import numpy as np
import tensorflow as tf
import os



def main():

    data = np.load(os.path.join(
        "npy","all_jet_images.npy"
    ),allow_pickle=True)

    train_model(data,epochs=500,steps=50,mini_batch_size=40)






if __name__ == "__main__":

    main()
