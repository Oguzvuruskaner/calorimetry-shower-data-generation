import os

from src.training.IModel import IModel
from src.training.validation_tasks.IValidationTask import IValidationTask
import tensorflow as tf
import matplotlib.pyplot as plt


class JetImageComparison(IValidationTask):

    def __init__(self,root_dir : str):
        self._image_root_dir = root_dir

    def validate(self, models: [tf.keras.Model], data, results):

        autoencoder = models[-1]
        _dimension = int(data.shape[1]**.5)


        for ind,original in enumerate(data):

            predicted = autoencoder.predict(original.reshape(1,original.shape[0]))

            fig = plt.figure(dpi=200)
            fig.suptitle("Jet Comparison {}".format(ind), fontsize=36)
            grid_spec = fig.add_gridspec(2,4)
            fig.set_size_inches(20,10)

            original_plot = original.reshape((_dimension,_dimension))
            ax1 = fig.add_subplot(grid_spec[:,0:2])
            original_img = ax1.imshow(original_plot)
            ax1.set_title("Original Jet",fontsize=32)
            plt.colorbar(original_img, ax=ax1)

            predicted_plot = predicted.reshape((_dimension,_dimension))
            ax2 = fig.add_subplot(grid_spec[:,2:])
            predicted_img = ax2.imshow(predicted_plot)
            ax2.set_title("Autoencoded Jet",fontsize=32)
            plt.colorbar(predicted_img, ax=ax2)


            plt.savefig(
                os.path.join(self._image_root_dir,"jet_comparison_{}.png".format(ind))
            )
            plt.close(fig)




