import time
import os

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import trange

from Model import wasserstein_loss
from config import DIMENSION,__MODEL_VERSION__
from scripts.test_model import plot_jet_generator_train_results, generate_jet_images

NOISE_INPUT_SIZE = 100


def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


class ClipConstraint(tf.keras.constraints.Constraint):

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


KERNEL_CONSTRAINT = ClipConstraint(clip_value=0.02)
OPTIMIZER = tf.keras.optimizers.RMSprop(.0005)


def create_critic() -> tf.keras.Model:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(DIMENSION * DIMENSION,)),
        tf.keras.layers.Reshape((DIMENSION,DIMENSION,1)),

        tf.keras.layers.ZeroPadding2D((2,2)),
        tf.keras.layers.LocallyConnected2D(9,(5,5)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.ZeroPadding2D((2,2)),
        tf.keras.layers.LocallyConnected2D(9, (5,5)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.4),


        tf.keras.layers.LocallyConnected2D(8,(3,3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.AveragePooling2D(),

        tf.keras.layers.LocallyConnected2D(8, (3, 3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.AveragePooling2D(),

        tf.keras.layers.Flatten(),


        tf.keras.layers.Dense(1,activation="tanh",kernel_constraint=KERNEL_CONSTRAINT)

    ],name="v{}_jet_critic".format(__MODEL_VERSION__))


    model.summary()

    return model


def create_generator(noise_input_size = NOISE_INPUT_SIZE) -> tf.keras.Model:


    model = tf.keras.Sequential([
        tf.keras.Input(shape=(noise_input_size,)),

        tf.keras.layers.Dense(DIMENSION * DIMENSION * 2),
        tf.keras.layers.Reshape((DIMENSION//4,DIMENSION//4,32)),

        tf.keras.layers.Conv2D(32,(5,5),padding="same",kernel_constraint=KERNEL_CONSTRAINT),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.UpSampling2D((2,2)),

        tf.keras.layers.ZeroPadding2D((2,2)),
        tf.keras.layers.LocallyConnected2D(6,(5,5),kernel_constraint=KERNEL_CONSTRAINT),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.UpSampling2D(),

        tf.keras.layers.ZeroPadding2D((2,2)),
        tf.keras.layers.LocallyConnected2D(6,(5,5),kernel_constraint=KERNEL_CONSTRAINT),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.LocallyConnected2D(6,(5,5),kernel_constraint=KERNEL_CONSTRAINT),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.ZeroPadding2D((2, 2)),
        tf.keras.layers.LocallyConnected2D(1, (5, 5), activation="sigmoid"),

        tf.keras.layers.Flatten()

    ],name="v{}_jet_generator".format(__MODEL_VERSION__))

    model.summary()

    return model

def create_gan(critic:tf.keras.Model,generator:tf.keras.Model) -> (
    tf.keras.Model,
    tf.keras.Model,
    tf.keras.Model,
):

    generator_train_input = tf.keras.Input(shape=(NOISE_INPUT_SIZE,))
    critic_train_input = tf.keras.Input(shape=(NOISE_INPUT_SIZE,))
    critic_real_input = tf.keras.Input(shape=(DIMENSION*DIMENSION))

    for layer in critic.layers:
        layer.trainable = True
    critic.trainable = True

    for layer in generator.layers:
        layer.trainable = False
    generator.trainable = False

    critic_real = tf.keras.Model(
        inputs = [critic_real_input],
        outputs = [critic(critic_real_input)]
    )
    critic_real.compile(OPTIMIZER,wasserstein_loss)


    critic_gan = tf.keras.Model(
        inputs=[
            critic_train_input
        ],
        outputs=[
            critic(generator(critic_train_input))
        ],
        name="Critic_GAN_{}".format(__MODEL_VERSION__)
    )
    critic_gan.compile(OPTIMIZER,wasserstein_loss)

    for layer in critic.layers:
        layer.trainable = False
    critic.trainable = False

    for layer in generator.layers:
        layer.trainable = True
    generator.trainable = True

    generator_gan = tf.keras.Model(
        inputs=[
            generator_train_input
        ],
        outputs=[
            critic(generator(generator_train_input))
        ],
        name="Generator_GAN_{}".format(__MODEL_VERSION__)
    )
    generator_gan.compile(OPTIMIZER,wasserstein_loss)


    critic_real.summary()
    critic_gan.summary()
    generator_gan.summary()

    return (critic_real,critic_gan,generator_gan)


def train_model(data,epochs=200,steps = 500,mini_batch_size=50):


    generator = create_generator()
    critic = create_critic()
    critic_real,real_gan,fake_gan = create_gan(critic,generator)


    data = data.reshape((len(data),DIMENSION*DIMENSION))


    x_train,x_test,_,_ = train_test_split(
        data,
        np.ones((len(data),1)),
        test_size=.1,
        shuffle=True,
        random_state=int(time.time())
    )

    epoch_losses = np.zeros((epochs,4))
    step_losses = np.zeros((steps,3))

    for epoch in trange(epochs):

        for step in range(steps):

            critic_fake_input = np.random.normal(size=(mini_batch_size,NOISE_INPUT_SIZE))
            generator_predict_input = np.random.normal(size=(mini_batch_size//5,NOISE_INPUT_SIZE))

            step_losses[step, 0] = critic_real.train_on_batch(
                x_train[np.random.choice(len(x_train), mini_batch_size)],
                # This is equivalent to np.random.randint(0,len(x_train),3)
                -np.ones((mini_batch_size, 1))
            )


            step_losses[step,1] = real_gan.train_on_batch(critic_fake_input,np.ones((mini_batch_size,1)))
            step_losses[step,2] = fake_gan.train_on_batch(generator_predict_input,-np.ones((mini_batch_size//5,1)))


        epoch_losses[epoch,:3] = np.mean(step_losses.T,axis=1)
        epoch_losses[epoch,3] = critic_real.evaluate(x_test,-np.ones((len(x_test),1)),verbose=0)

    tf.keras.utils.plot_model(generator, os.path.join("models", "{}_arch.png".format(generator.name)), show_shapes=True)
    generator.save(os.path.join("models", "{}.hdf5".format(generator.name)))

    tf.keras.utils.plot_model(critic, os.path.join("models", "{}_arch.png".format(critic.name)), show_shapes=True)
    critic.save(os.path.join("models", "{}.hdf5".format(critic.name)))


    plot_jet_generator_train_results(
        epoch_losses,
        os.path.join("models","jet_generator_train_results_{}.png".format(__MODEL_VERSION__))
    )

    generate_jet_images(generator)

