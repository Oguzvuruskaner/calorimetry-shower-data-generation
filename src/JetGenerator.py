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


class ClipConstraint(tf.keras.constraints.Constraint):

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}



KERNEL_INITIALIZER = tf.keras.initializers.glorot_uniform(seed=int(time.time()))
KERNEL_CONSTRAINT = ClipConstraint(clip_value=.02)
OPTIMIZER = tf.keras.optimizers.RMSprop(0.0005)
DROPOUT_RATE = .25
LOSS = wasserstein_loss
LATENT_SIZE = 100

CUSTOM_OBJECTS = {
    "KERNEL_INITIALIZER":KERNEL_INITIALIZER,
    "KERNEL_CONSTRAINT":KERNEL_CONSTRAINT,
    "OPTIMIZER":OPTIMIZER
}


_Dense = lambda output_size,activation=None: tf.keras.layers.Dense(
    output_size,
    kernel_constraint=KERNEL_CONSTRAINT,
    kernel_initializer=KERNEL_INITIALIZER,
    kernel_regularizer=tf.keras.regularizers.l1(),
    bias_regularizer=tf.keras.regularizers.l2(),
    activation=activation
)

_Conv2d = lambda filters,kernel_size,activation=None : tf.keras.layers.Conv2D(
    filters=filters,
    kernel_size=kernel_size,
    kernel_initializer=KERNEL_INITIALIZER,
    kernel_constraint=KERNEL_CONSTRAINT,
    kernel_regularizer=tf.keras.regularizers.l1(),
    bias_regularizer=tf.keras.regularizers.l2(),
    padding="same",
    activation=activation
)

_LocallyConnected2D = lambda filters,kernel_size,activation=None: tf.keras.layers.LocallyConnected2D(
    filters = filters,
    kernel_size=kernel_size,
    kernel_initializer=KERNEL_INITIALIZER,
    kernel_constraint=KERNEL_CONSTRAINT,
    kernel_regularizer=tf.keras.regularizers.l1(),
    bias_regularizer=tf.keras.regularizers.l2(),
    activation=activation
)



def get_latent_input(batch_size=64):
    return np.random.normal(size=(batch_size, LATENT_SIZE))


def create_critic() -> tf.keras.Model:

    _dimension = int(sqrt(N_COMPONENTS))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(N_COMPONENTS)),

        _Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        _Dense(N_COMPONENTS),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Reshape((_dimension,_dimension,1)),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.LocallyConnected2D(9, (5,5)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.LocallyConnected2D(1,(5,5)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        _Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        _Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),



        _Dense(1,activation="tanh")

    ],name="v{}_jet_critic".format(__MODEL_VERSION__))


    model.summary()

    return model


def create_generator() -> tf.keras.Model:

    _dimension = int(sqrt(N_COMPONENTS))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(LATENT_SIZE),

        _Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        _Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        _Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        _Dense(N_COMPONENTS),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Reshape((_dimension, _dimension, 1)),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.LocallyConnected2D(7, (5, 5)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.LocallyConnected2D(12, (5, 5)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ZeroPadding2D((2,2)),
        tf.keras.layers.LocallyConnected2D(1, (5,5)),


        tf.keras.layers.Flatten(),

        _Dense(N_COMPONENTS),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization()

    ],name="v{}_jet_generator".format(__MODEL_VERSION__))

    model.summary()

    return model


def train_model(
        data,
        epochs=200,
        steps = 500,
        batch_size=1024,
        save_results = True,
        generator_model=None,
        critic_model=None
        ):

    pca = PCA(N_COMPONENTS)

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

    data = data.reshape((len(data),DIMENSION*DIMENSION))



    root_dir = os.path.join("results","pca_images_{}".format(__MODEL_VERSION__))

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)


    data = pca.fit_transform(data)
    pca_results = pca.inverse_transform(data)

    for ind,result in enumerate(pca_results,start=1):
        image = result.reshape((DIMENSION,DIMENSION)) * 50

        save_jet_image(image,os.path.join(root_dir, "{}.png".format(ind)))

    del pca_results

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

            for i in range(5):


                step_losses[step, 0] = critic.train_on_batch(
                    x_train[np.random.randint(0,len(data),batch_size)],
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

        generate_jet_images(generator,pca=pca)


        plot_jet_generator_train_results(
            epoch_losses,
            os.path.join("models", "jet_generator_train_results_{}.png".format(__MODEL_VERSION__))
        )


    return generator,critic,epoch_losses


def load_models():

    critic = tf.keras.models.load_model()
    generator =  tf.keras.models.load_model()

    return generator,critic