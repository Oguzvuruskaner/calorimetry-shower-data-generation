import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.colors import LogNorm
from tqdm import tqdm

from src.config import PLOT_ERROR


def plot_data(data : np.array,title,ax=None):

    from src.config import HIT_Z_MIN, HIT_Z_MAX,HIT_R_MAX

    if ax == None:
        fig = plt.figure()
        ax : plt.Axes = fig.add_subplot(111)

    axes_image = ax.imshow(
        data+PLOT_ERROR,

        extent=[HIT_Z_MIN,HIT_Z_MAX,HIT_R_MAX,0]
    )

    ax.set_xlabel("Z")
    ax.set_ylabel("R")
    ax.set_title(title)

    plt.colorbar(axes_image)

    return ax


def plot_multiple_images(data,nrow : int):

    ncolumn = len(data) // nrow
    remainder = len(data) % nrow
    counter = 0

    sup_figure = plt.figure(dpi=200)
    sup_figure.set_size_inches(ncolumn*5,(nrow+1)*5)
    sup_figure.subplots_adjust(wspace=0.75,hspace=0.75)
    grid_spec = sup_figure.add_gridspec(nrow, ncolumn)


    for i in range(nrow-1):
        for j in range(ncolumn):
            ax = sup_figure.add_subplot(grid_spec[i, j])
            plot_data(data[i*ncolumn + j],counter,ax=ax)
            counter += 1

    for i in range(remainder):
        ax = sup_figure.add_subplot(grid_spec[nrow-1, i])
        plot_data(data[(nrow-1) * ncolumn + i], counter, ax=ax)
        counter +=1

    return sup_figure


def plot_images(data,root_directory:str):

    for ind,img in enumerate(tqdm(data)):
        IMAGE_PATH = os.path.join(root_directory,"{}.png".format(ind))
        plot_data(img,ind)
        plt.savefig(IMAGE_PATH)
        plt.close()


