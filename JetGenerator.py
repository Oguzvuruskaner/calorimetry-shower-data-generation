import time

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Model import wasserstein_loss
from config import DIMENSION,__MODEL_VERSION__

NOISE_INPUT_SIZE = 100




def create_critic() -> tf.keras.Model:

    model = tf.keras.models.Sequential({
        tf.keras.layers.Input(shape=(DIMENSION * DIMENSION,)),

    })

    for i in range(4):
        tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        tf.keras.layers.LocallyConnected2D(16*2**i, (3, 3))
        tf.keras.layers.LeakyReLU()
        tf.keras.layers.MaxPooling2D()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(DIMENSION))
    model.add(tf.keras.layers.Dense(1, activation="tanh"))


    return model


def create_generator(noise_input_size = NOISE_INPUT_SIZE,upsampling_count = 4) -> tf.keras.Model:




    model = tf.keras.Sequential([
        tf.keras.Input(shape=(noise_input_size,)),
        tf.keras.layers.Dense(DIMENSION*DIMENSION//4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((DIMENSION // upsampling_count, DIMENSION // upsampling_count)),

    ])

    for i in range(upsampling_count):
        tf.keras.layers.UpSampling2D(interpolation="bilinear"),
        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.LocallyConnected2D(32, (3, 3)),


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(DIMENSION*DIMENSION,activation="sigmoid"))

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

    critic_real = critic(critic_real_input)

    critic_real.name = "Critic_{}".format(__MODEL_VERSION__)
    critic_real.compile("adam",wasserstein_loss)


    critic_gan = tf.keras.Model(
        inputs=[
            critic_train_input
        ],
        outputs=[
            critic(generator(critic_train_input))
        ],
        name="Critic_GAN_{}".format(__MODEL_VERSION__)
    )
    critic_gan.compile("adam",wasserstein_loss)

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




    return (critic_real,critic_gan,generator_gan)

def train_model(data,epochs=200,steps = 500,mini_batch_size=25):


    generator = create_generator()
    critic = create_critic()
    critic_real,real_gan,fake_gan = create_gan(critic,generator)


    data = np.reshape((len(data),DIMENSION*DIMENSION))


    x_train,x_test,_,_ = train_test_split(
        data,
        np.ones((len(data),1)),
        test_size=.3,
        shuffle=True,
        random_state=time.time()
    )

    epoch_losses = np.zeros((epochs,4))
    step_losses = np.zeros((steps,3))

    for epoch in tqdm(range(epochs)):

        for step in range(steps):

            critic_fake_input = np.random.normal((mini_batch_size,NOISE_INPUT_SIZE))
            generator_predict_input = np.random.normal((mini_batch_size//5,NOISE_INPUT_SIZE))

            step_losses[step,0] = critic_real.train_on_batch(
                np.random.choice(x_train,mini_batch_size),
                np.ones((mini_batch_size,1)))

            step_losses[step,1] = real_gan.train_on_batch(critic_fake_input,-np.ones((mini_batch_size,1)))

            step_losses[step,2] = fake_gan.train_on_batch(generator_predict_input,np.ones((mini_batch_size,1)))


        epoch_losses[epoch,:3] = np.mean(step_losses,axis=1)
        epoch_losses[epoch,4] = real_gan.evaluate(x_test,np.ones((x_test,1)))





