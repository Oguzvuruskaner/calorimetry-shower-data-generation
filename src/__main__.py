import os

import tensorflow as tf
import numpy as np

from src.datasets.JetImageDataset import JetImageDataset
from src.side_effects.PlotClosest import PlotClosest
from src.side_effects.WriteImages import WriteImages
from src.transformers.Flatten import Flatten
from src.scripts.problems import train_jet_generator
from src.config import DIMENSION, __MODEL_VERSION__,N_COMPONENTS
from sklearn.decomposition import PCA
from src.config import DIMENSION

if __name__ == "__main__":



    dataset = JetImageDataset(dimension=DIMENSION) \
        .set_np_path(os.path.join("npy", "jet_array_{}x{}.npy".format(DIMENSION, DIMENSION))) \
        .obtain(from_npy=True) \
        .add_pre_transformation(Flatten()) \
        .apply_pre_transformations()\

    data = dataset.array()



    generator,critic,_ = train_jet_generator(data)


    predictions = generator.predict(np.random.normal(0,1,(200,1024)))

    predictions.resize((predictions.shape[0],DIMENSION,DIMENSION))

    original_images_path = os.path.join("jet_images","{}x{}_images".format(DIMENSION,DIMENSION))
    predicted_images_path = os.path.join("results","{}x{}_results".format(DIMENSION,DIMENSION))

    if not os.path.exists(original_images_path):
        os.mkdir(original_images_path)

    if not os.path.exists(predicted_images_path):
        os.mkdir(predicted_images_path)

    WriteImages(original_images_path).transform(data.reshape((data.shape[0],DIMENSION,DIMENSION)))
    WriteImages(predicted_images_path).transform(predictions)

    plot = PlotClosest(predictions,os.path.join("results","image_comparison_{}".format(__MODEL_VERSION__)))
    plot.transform(data.resize((data.shape[0],DIMENSION,DIMENSION)))



