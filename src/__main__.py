import os

import tensorflow as tf
import numpy as np

from src.datasets.JetImageDataset import JetImageDataset
from src.scripts.test_model import save_jet_image, generate_jet_images
from src.side_effects.PlotClosest import PlotClosest
from src.training.models.autoencoders.JetImageCompressor import JetImageCompressor
from src.training.validation_tasks.JetImageComparison import JetImageComparison
from src.transformers.Flatten import Flatten
from src.scripts.problems import train_jet_generator
from src.config import DIMENSION, __MODEL_VERSION__,N_COMPONENTS
from sklearn.decomposition import PCA

if __name__ == "__main__":

    pca = PCA(N_COMPONENTS*N_COMPONENTS)

    dataset = JetImageDataset() \
        .set_np_path(os.path.join("npy", "jet_array_{}x{}.npy".format(DIMENSION,DIMENSION))) \
        .obtain(from_npy=True) \
        .add_pre_transformation(Flatten())\
        .apply_pre_transformations()\

    data = dataset.array()

    transformed_data = pca.fit_transform(data)


    generator,critic,_ = train_jet_generator(transformed_data)


    predictions = generator.predict(np.random.normal(0,1,(200,1024)))
    inverted_predictions = pca.inverse_transform(predictions)

    inverted_predictions.resize((predictions.shape[0],DIMENSION,DIMENSION))

    for pred_no,pred in enumerate(inverted_predictions):
        save_jet_image(pred,os.path.join("results","jet_images_{}".format(__MODEL_VERSION__),"{}.png".format(pred_no)))



    plot = PlotClosest(inverted_predictions)
    plot.transform(data.resize((data.shape[0],DIMENSION,DIMENSION)))



    generate_jet_images(generator,pca=pca)


