import os

import numpy as np

from src.JetGenerator import train_model
from src.scripts.test_model import plot_jet_generator_train_results
from src.config import __MODEL_VERSION__



EPOCHS = 75
STEPS_PER_EPOCH = 100






def easy_problem():

    # Create images with dim=DIMENSION
    # Each image has random real numbers inside it summing up to 1.


    from src.config import DIMENSION

    TOTAL_IMAGES = 10000

    data = np.random.normal(0,1,(TOTAL_IMAGES,DIMENSION*DIMENSION,1))
    data /= data.sum(axis=1,keepdims=1)

    data_sums = data.sum(axis=1)

    print("Mean: {}".format(np.mean(data_sums)))
    print("Std: {}".format(np.std(data_sums)))


    data.resize((TOTAL_IMAGES,DIMENSION,DIMENSION))

    generator,critic,epoch_losses = train_model(data,epochs=EPOCHS,steps=STEPS_PER_EPOCH,save_results=False)

    predictions = generator.predict(np.random.normal(size=(200,100)))
    sums = predictions.sum(axis=1)

    print("Mean: {}".format(np.mean(sums)))
    print("Std: {}".format(np.std(sums)))

    plot_jet_generator_train_results(
        epoch_losses,
        os.path.join("easy_problem.png")
    )

    print("Critic Result: {}".format(critic.predict(np.zeros(1,DIMENSION,DIMENSION))))


def very_easy_problem():
    # Create images with dim=DIMENSION
    # Each image has zeros.

    from src.config import DIMENSION

    TOTAL_IMAGES = 10000

    data = np.zeros((TOTAL_IMAGES, DIMENSION, DIMENSION))

    generator, critic, epoch_losses = train_model(data,epochs=EPOCHS,steps=STEPS_PER_EPOCH, save_results=False)

    predictions = generator.predict(np.random.normal(size=(200, 100)))
    sums = predictions.sum(axis=1)

    print("Mean: {}".format(np.mean(sums)))
    print("Std: {}".format(np.std(sums)))

    plot_jet_generator_train_results(
        epoch_losses,
        os.path.join("very_easy_problem.png")
    )

    print("Critic Result: {}".format(
        critic.predict(
            np.zeros((1,DIMENSION*DIMENSION))
        )
    ))

    return generator,critic,epoch_losses


def train_jet_generator(data=None):

    if data is None:
        data = np.load(os.path.join(
            "npy","all_jet_images.npy"
        ),allow_pickle=True)

    return train_model(data,epochs=EPOCHS,steps=STEPS_PER_EPOCH)