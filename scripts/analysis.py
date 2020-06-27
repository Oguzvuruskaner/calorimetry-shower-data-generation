import os

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def plot_cum_pca():

    data = np.load(
        os.path.join("npy","all_jet_images.npy")
    )

    data.resize((data.shape[0],data.shape[1] * data.shape[2]))

    pca = PCA()

    # n_components <= min(num_samples,num_features)
    total_indices = min(data.shape[0],data.shape[1])


    pca.n_components = total_indices
    pca.fit(data)

    variance_array= np.cumsum(pca.explained_variance_ratio_)


    plt.title("PCA Results of jet images")
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Number of Components")

    plt.plot(np.arange(total_indices),variance_array)

    plt.savefig(
        os.path.join("plots","pca_compression_jet_images.png")
    )
