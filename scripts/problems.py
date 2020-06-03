import os

import numpy as np

from JetGenerator import train_model
from scripts.test_model import plot_jet_generator_train_results


def easy_problem():

    # Create images with dim=DIMENSION
    # Each image has random real numbers inside it summing up to 1.


    from config import DIMENSION

    TOTAL_IMAGES = 10000

    data = np.random.normal(0,1,(TOTAL_IMAGES,DIMENSION*DIMENSION,1))
    data /= data.sum(axis=1,keepdims=1)

    data_sums = data.sum(axis=1)

    print("Mean: {}".format(np.mean(data_sums)))
    print("Std: {}".format(np.std(data_sums)))


    data.resize((TOTAL_IMAGES,DIMENSION,DIMENSION))

    generator,critic,epoch_losses = train_model(data,epochs=100,steps=20,mini_batch_size=60,save_results=False)

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

    from config import DIMENSION

    TOTAL_IMAGES = 10000

    data = np.zeros((TOTAL_IMAGES, DIMENSION, DIMENSION))

    generator, critic, epoch_losses = train_model(data, epochs=100, steps=20, mini_batch_size=60, save_results=False)

    predictions = generator.predict(np.random.normal(size=(200, 100)))
    sums = predictions.sum(axis=1)

    print("Mean: {}".format(np.mean(sums)))
    print("Std: {}".format(np.std(sums)))

    plot_jet_generator_train_results(
        epoch_losses,
        os.path.join("easy_problem.png")
    )

    print("Critic Result: {}".format(critic.predict(np.zeros(1,DIMENSION,DIMENSION))))


def train_jet_generator():

    data = np.load(os.path.join(
        "npy","all_jet_images.npy"
    ),allow_pickle=True)

    train_model(data,epochs=150,steps=75,mini_batch_size=60)