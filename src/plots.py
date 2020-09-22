import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np



def plot_data(data,title:str,ax=None):

    from src.config import HIT_Z_MIN, HIT_Z_MAX,HIT_E_MAX,HIT_R_MAX

    if ax == None:

        fig = plt.figure()
        ax : plt.Axes = fig.add_subplot(111)

    axes_image = ax.imshow(
        data * 50,
        norm=colors.LogNorm(vmax=HIT_E_MAX),
        extent=[HIT_Z_MIN,HIT_Z_MAX,HIT_R_MAX,0]
    )
    axes_image.cmap.set_bad(color=axes_image.cmap(10e-20))
    ax.set_xlabel("Z")
    ax.set_ylabel("R")
    ax.set_title(title)

    if ax == None:
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
            plot_data(data[i*ncolumn + j]+1e-40,counter,ax=ax)
            counter += 1
            sup_figure.colorbar(ax.images[0])

    for i in range(remainder):
        ax = sup_figure.add_subplot(grid_spec[nrow-1, i])
        plot_data(data[(nrow-1) * ncolumn + i]+1e-40, counter, ax=ax)
        sup_figure.colorbar(ax.images[0])

    return sup_figure




