import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from config import N_COMPONENTS


class PCALayer(tf.keras.layers.Layer):

    def __init__(self,pca:PCA=None):
        super(PCALayer,self).__init__()
        self._pca = pca



class PCAEncoder(PCALayer):

    def call(self, inputs, **kwargs):
        if not self._pca:
            return inputs

        return self._pca.transform(inputs)



class PCADecoder(PCALayer):

    def call(self, inputs, **kwargs):
        if not self._pca:
            return inputs

        return self._pca.inverse_transform(inputs)




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

    return pca,updated_data


