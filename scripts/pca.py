import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.decomposition import PCA
from config import N_COMPONENTS



def plot_cum_pca():

    data = np.load(
        os.path.join("npy","all_jet_images.npy")
    )

    data.resize((data.shape[0],data.shape[1] * data.shape[2]))

    # n_components <= min(num_samples,num_features)
    total_indices = min(data.shape[0],data.shape[1])

    pca = PCA(total_indices)



    pca.fit(data)

    variance_array= np.cumsum(pca.explained_variance_ratio_)


    plt.title("PCA Results of jet images")
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Number of Components")

    plt.plot(np.arange(total_indices),variance_array)

    plt.savefig(
        os.path.join("plots","pca_compression_jet_images.png")
    )

# With respect to 1200 jet images data,
# using 4-component PCA explains 0.99982353 variance of data
# which is sufficient.

def create_pca_compression():

    pca = PCA(N_COMPONENTS)

    data = np.load(
        os.path.join("npy", "all_jet_images.npy")
    )

    data.resize((data.shape[0], data.shape[1] * data.shape[2]))



    updated_data = pca.fit_transform(data)

    np.save(
        os.path.join("npy","{}_component_jet_images".format(N_COMPONENTS)),
        updated_data
    )

    with open(
        os.path.join("npy","{}_component_pca".format(N_COMPONENTS)),
        "wb"
    ) as fp:

        pickle.dump(pca,fp)



