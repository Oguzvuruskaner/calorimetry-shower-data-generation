from AdvancedModel import train_model

import numpy as np
import os

from scripts.scripts import scale_jets, get_root_files
from readRoot import create_jet_images



def main():

    create_jet_images(get_root_files())
    data = np.load(
        os.path.join(
            "npy",
            "all_jet_images.npy"
        ),
        allow_pickle=True

    )
    train_model()


if __name__ == "__main__":

    main()
