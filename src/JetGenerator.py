import time
import os
from math import sqrt

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import trange

from src.config import DIMENSION,__MODEL_VERSION__,N_COMPONENTS
from src.scripts.pca import write_pca_to_csv
from src.scripts.test_model import plot_jet_generator_train_results, generate_jet_images, save_jet_image

from sklearn.cluster import KMeans

def continue_training(model_version=__MODEL_VERSION__):



    critic = tf.keras.models.load_model(
        os.path.join("models","v{}_jet_critic.hdf5".format(model_version))
    )
    generator = tf.keras.models.load_model(
        os.path.join("models", "v{}_jet_generator.hdf5".format(model_version)),
    )


    return critic,generator

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(-y_true * y_pred)

DROPOUT_RATE = .25
LATENT_SIZE = 1024
OPTIMIZER = tf.keras.optimizers.Adam(0.1)


def get_latent_input(batch_size=64):
    return np.random.normal(size=(batch_size, LATENT_SIZE))


def create_critic() -> tf.keras.Model:


    input_layer = tf.keras.layers.Input(N_COMPONENTS*N_COMPONENTS)

    x = tf.keras.layers.Reshape((N_COMPONENTS,N_COMPONENTS,1))(input_layer)

    x = tf.keras.layers.ZeroPadding2D((2,2))(x)
    x = tf.keras.layers.LocallyConnected2D(12,(5,5))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ZeroPadding2D((2, 2))(x)
    x = tf.keras.layers.LocallyConnected2D(7, (5, 5))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ZeroPadding2D((2, 2))(x)
    x = tf.keras.layers.LocallyConnected2D(1, (5, 5))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(1,activation="tanh")(x)

    model = tf.keras.Model(input_layer,output_layer,name="Critic_{}".format(__MODEL_VERSION__))

    model.summary()

    return model


def create_generator() -> tf.keras.Model:



    input_layer = tf.keras.layers.Input(LATENT_SIZE)

    x = tf.keras.layers.Dense(N_COMPONENTS*N_COMPONENTS)(input_layer)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(.4)(x)

    x = tf.keras.layers.Reshape((N_COMPONENTS,N_COMPONENTS,1))(x)

    x = tf.keras.layers.ZeroPadding2D((2, 2))(x)
    x = tf.keras.layers.LocallyConnected2D(12, (5, 5))(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.ZeroPadding2D((2, 2))(x)
    x = tf.keras.layers.LocallyConnected2D(7, (5, 5))(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.ZeroPadding2D((2, 2))(x)
    x = tf.keras.layers.LocallyConnected2D(1, (5, 5))(x)
    x = tf.keras.layers.LeakyReLU()(x)


    output_layer = tf.keras.layers.Flatten()(x)



    model = tf.keras.Model(input_layer, output_layer,name="Generator_{}".format(__MODEL_VERSION__))

    model.summary()


    return model


def train_model(
        data,
        epochs=200,
        steps = 500,
        batch_size=32,
        save_results = True,
        generator_model=None,
        critic_model=None
    ):


    if generator_model:
        generator = generator_model
    else:
        generator = create_generator()

    if critic_model:
        critic = critic_model
    else:
        critic = create_critic()


    critic_fake_input = tf.keras.Input(LATENT_SIZE)
    critic_fake_gan = tf.keras.Model(critic_fake_input, critic(generator(critic_fake_input)))
    critic_fake_gan.compile(OPTIMIZER,wasserstein_loss)

    generator_train_input = tf.keras.Input(LATENT_SIZE)
    generator_gan = tf.keras.Model(generator_train_input, critic(generator(generator_train_input)))
    generator_gan.compile(OPTIMIZER,wasserstein_loss)

    critic.compile(OPTIMIZER,wasserstein_loss)

    epoch_losses = np.zeros((epochs,4))
    step_losses = np.zeros((steps,3))

    for epoch in trange(epochs):

        x_train, x_test, _, _ = train_test_split(
            data,
            np.ones((len(data), 1)),
            test_size=.1,
            shuffle=True,
            random_state=int(time.time())
        )

        for step in range(steps):

            critic.trainable = True
            generator.trainable = False

            for i in range(4):


                step_losses[step, 0] = critic.train_on_batch(
                    x_train[np.random.randint(0,len(x_train),batch_size)],
                    np.ones((batch_size, 1))
                )

                step_losses[step,1] = critic_fake_gan.train_on_batch(np.random.normal(size=(batch_size,LATENT_SIZE)),-np.ones((batch_size,1)))

            critic.trainable = False
            generator.trainable = True

            step_losses[step,2] = generator_gan.train_on_batch(np.random.normal(size=(batch_size,LATENT_SIZE)),np.ones((batch_size,1)))


        epoch_losses[epoch,:3] = np.mean(step_losses.T,axis=1)
        epoch_losses[epoch,3] = critic.evaluate(x_test,np.ones((len(x_test),1)),verbose=0)

    if save_results:

        tf.keras.utils.plot_model(generator, os.path.join("models", "{}_arch.png".format(generator.name)), show_shapes=True)
        generator.save(os.path.join("models", "{}.hdf5".format(generator.name)))

        tf.keras.utils.plot_model(critic, os.path.join("models", "{}_arch.png".format(critic.name)), show_shapes=True)
        critic.save(os.path.join("models", "{}.hdf5".format(critic.name)))

        plot_jet_generator_train_results(
            epoch_losses,
            os.path.join("models", "jet_generator_train_results_{}.png".format(__MODEL_VERSION__))
        )


    return generator,critic,epoch_losses


def load_models():

    critic = tf.keras.models.load_model()
    generator =  tf.keras.models.load_model()

    return generator,critic