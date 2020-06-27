import tensorflow as tf
import numpy as np
from tqdm import trange

LATENT_SIZE = 100



def create_generator():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(LATENT_SIZE),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Reshape((32,32,1)),

        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.LocallyConnected2D(7,(3,3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.LocallyConnected2D(5,(3,3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.LocallyConnected2D(1,(3,3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten()

    ])


    return model


def train_critic():

    critic = create_critic()

    critic.compile(loss="mse")



def train_model(epochs=100,steps_per_epoch=100):

    generator = create_generator()
    critic = create_critic()




    critic_fake_input = tf.keras.Input(LATENT_SIZE)
    critic_fake_gan = tf.keras.Model(critic_fake_input,critic(generator(critic_fake_input)))
    critic_fake_gan.compile(loss=wasserstein_loss)


    generator_train_input = tf.keras.Input(LATENT_SIZE)
    generator_gan = tf.keras.Model(generator_train_input,critic(generator(generator_train_input)))
    generator_gan.compile(loss=wasserstein_loss)


    critic.compile(loss=wasserstein_loss)
    data = np.zeros((5,1024))
    correct_label = np.ones((5,1))
    fake_label = -np.ones((5,1))


    for epoch_number in trange(epochs):

        for step_number in range(steps_per_epoch):

            critic.trainable = True
            generator.trainable = False

            critic.train_on_batch(data,correct_label)
            critic_fake_gan.train_on_batch(np.random.normal(0,1,(5,100)),fake_label)

            generator.trainable = True
            critic.trainable = False

            generator_gan.train_on_batch(np.random.normal(0,1,(1,100)),np.ones((1,1)))


    results = generator.predict(np.random.normal(0,1,(100,100)))



def wasserstein_loss(y_pred,y_true):
    return tf.keras.backend.mean(-y_pred * y_true)


def create_critic():


    model = tf.keras.Sequential([
        tf.keras.layers.Input(1024),
        tf.keras.layers.Reshape((32,32,1)),

        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.LocallyConnected2D(5,(3,3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ZeroPadding2D((1, 1)),
        tf.keras.layers.LocallyConnected2D(5, (3, 3)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Dense(1,activation="tanh")

    ])


    return model