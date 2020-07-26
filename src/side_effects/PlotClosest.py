import os

import numpy as np

from src.side_effects.ISideEffect import ISideEffect
import matplotlib.pyplot as plt



class PlotClosest(ISideEffect):

    def __init__(self,plot_list : np.array,root_directory:str):
        self._plot_list = plot_list
        self._directory_path = root_directory

    def transform(self, data):

        # Data should be 3-dimensional array.

        for ind,img in enumerate(self._plot_list):
            min_score = float("inf")
            min_distance_image = None

            for origImage in data:

                current_score =  np.sum((img - origImage)**2)
                if current_score < min_score:
                    min_distance_image = np.copy(origImage)


            fig : plt.Figure = plt.figure(constrained_layout = False)
            fig.set_size_inches(60,20)

            grid_spec = fig.add_gridspec(1,3)

            original_axes = fig.add_subplot(grid_spec[0,0])
            original_axes.set_title("Closest Image",fontsize=40)
            original_axes.imshow(min_distance_image,vmax=1,vmin=0,cmap="gray")

            generated_axes = fig.add_subplot(grid_spec[0,2])
            generated_axes.set_title("Generated Image",fontsize=40)
            generated_axes.imshow(img,vmax = 1,vmin = 0,cmap="gray")

            fig.savefig(os.path.join(self._directory_path,"{}.png".format(ind)))
            plt.close(fig)


    def __str__(self):
        return "Plot Closest"