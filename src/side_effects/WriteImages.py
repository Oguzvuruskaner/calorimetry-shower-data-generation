import os

import matplotlib.pyplot as plt

from src.side_effects.ISideEffect import ISideEffect


class WriteImages(ISideEffect):

    def __init__(self,root_directory:str):

        self._directory = root_directory

    def transform(self, data):

        for ind,img in enumerate(data):
            file_path = os.path.join(self._directory,"{}.png".format(ind))
            plt.imsave(file_path,img,vmax=0,vmin=1,cmap="gray")

    def __str__(self):
        return "Write Images"



