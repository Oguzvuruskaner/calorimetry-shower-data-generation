import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.datasets.JetImageDataset import JetImageDataset
from src.side_effects.PlotClosest import PlotClosest
from src.side_effects.WriteImages import WriteImages
from src.transformers.Flatten import Flatten
from src.scripts.problems import train_jet_generator
from src.config import DIMENSION, __MODEL_VERSION__,LATENT_SIZE
from src.config import DIMENSION

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset = JetImageDataset(root_directories=[os.path.join("root_files")],dimension=32) \
        .set_store_path(os.path.join("npy", "jet_array_{}x{}.npy".format(32, 32 )))\
        .set_np_path(os.path.join("npy", "jet_array_{}x{}.npy".format(32, 32))) \
        .obtain(from_npy=True)

    data = dataset.array()


    data /= 50
    data.resize((data.shape[0],data.shape[1],data.shape[1]))
    _32_path = os.path.join("jet_images","32x32_images")

    for ind,img in enumerate(data):
        plt.imsave(os.path.join(_32_path,str(ind)+".png"),img.reshape(32,32),vmax=1,vmin=0,cmap="viridis")
