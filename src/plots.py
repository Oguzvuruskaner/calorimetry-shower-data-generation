import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

from src.config import PLOT_ERROR, MATRIX_DIMENSION
from src.config import HIT_Z_MIN, HIT_Z_MAX, HIT_R_MAX


def plot_data(data : np.array,title,ax=None):


    if ax == None:
        fig = plt.figure()
        ax : plt.Axes = fig.add_subplot(111)

    data[0] *= HIT_R_MAX
    data[1] *= HIT_Z_MAX

    axes_image = ax.imshow(
        (data)+PLOT_ERROR,
    )

    ax.set_xlabel("Z")
    ax.set_ylabel("R")
    ax.set_title(title)
    plt.colorbar(axes_image)
    return ax


def plot_energy_graph(data,title,energy,ax=None):

    if ax == None:
        fig = plt.figure()
        ax : plt.Axes = fig.add_subplot(111)

    ax.hist(data * energy,bins=32,range=(0,energy))
    ax.set_xlabel("E(GeV)")
    ax.set_ylabel("Number of Hits")
    ax.set_title(title)

    return ax



def plot_multiple_images(data,nrow : int,plot_func = plot_data):

    ncolumn = len(data) // nrow
    remainder = len(data) % nrow
    counter = 0

    sup_figure = plt.figure(dpi=200)
    sup_figure.set_size_inches(ncolumn*5,(nrow+1)*5)
    sup_figure.subplots_adjust(wspace=0.75,hspace=0.75)
    grid_spec = sup_figure.add_gridspec(nrow, ncolumn)

    if remainder == 0:
        iter_rows = nrow
    else:
        iter_rows = nrow - 1


    for i in range(iter_rows):
        for j in range(ncolumn):
            ax = sup_figure.add_subplot(grid_spec[i, j])
            plot_func(data[i*ncolumn + j],counter,ax=ax)
            counter += 1

    for i in range(remainder):
        ax = sup_figure.add_subplot(grid_spec[nrow-1, i])
        plot_func(data[(nrow-1) * ncolumn + i], counter, ax=ax)
        counter +=1

    return sup_figure



def plot_images(data,root_directory:str):

    for ind,img in enumerate(tqdm(data)):
        IMAGE_PATH = os.path.join(root_directory,"{}.png".format(ind))
        plot_data(img,ind)
        plt.savefig(IMAGE_PATH)
        plt.close()


def get_jet_images(data):

    images = np.zeros((len(data),MATRIX_DIMENSION,MATRIX_DIMENSION))

    for ind,jet in enumerate(data):

        images[ind] = np.histogramdd(jet[:, :2],
                             bins=MATRIX_DIMENSION,
                             range=np.array([[0, 1], [0, 1]]),
                             weights=jet[:, 2])[0]


    return images