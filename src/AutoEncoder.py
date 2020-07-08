import os

import numpy as np
import tensorflow as tf

from src.config import  __MODEL_VERSION__,DIMENSION,N_COMPONENTS
from src.datasets.JetImageDataset import JetImageDataset
from src.transformers.Flatten import Flatten
from src.transformers.Log10Scaling import Log10Scaling

from tqdm import trange



DROPOUT_RATE = 0.25
WHITE_NOISE_RATE = 0.01

BINARY_CROSSENTROPY = tf.keras.losses.BinaryCrossentropy()
KL_DIVERGENCE = tf.keras.losses.KLDivergence()

def custom_loss(y_true,y_pred):

    return BINARY_CROSSENTROPY(y_true,y_pred) + KL_DIVERGENCE(y_true,y_pred)

def create_decoder():

    model = tf.keras.Sequential([
        tf.keras.Input(N_COMPONENTS),

        tf.keras.layers.Dense(DIMENSION * DIMENSION // 16),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Reshape((DIMENSION//4,DIMENSION//4,1)),

        tf.keras.layers.Conv2DTranspose(64,(5,5),(2,2),padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Conv2DTranspose(32, (5, 5), (2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Conv2D(1,(5,5),padding="same",activation="sigmoid"),


        tf.keras.layers.Flatten()



    ],name="decoder_v{}".format(__MODEL_VERSION__))

    model.summary()

    return model


def create_encoder():

    model = tf.keras.Sequential([

        tf.keras.Input(DIMENSION*DIMENSION),

        tf.keras.layers.Dropout(WHITE_NOISE_RATE),
        tf.keras.layers.Reshape((DIMENSION,DIMENSION,1)),

        tf.keras.layers.Conv2D(32,(5,5),padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(64,(5,5),padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(1, (5, 5), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(N_COMPONENTS),
        tf.keras.layers.LeakyReLU(),


    ],name="encoder_v{}".format(__MODEL_VERSION__))

    model.summary()

    return model


def train_autoencoder(epochs = 300,steps=500,batch_size=64):

    data = JetImageDataset([os.path.join("root_files")]) \
        .set_np_path(os.path.join("npy", "jet_array_{}x{}.npy".format(DIMENSION, DIMENSION))) \
        .obtain(from_npy=True) \
        .add_invertible_transformation(Log10Scaling())\
        .add_pre_transformation(Flatten())\
        .apply_pre_transformations() \
        .array()


    encoder = create_encoder()
    decoder = create_decoder()

    autoencoder = tf.keras.Sequential([encoder,decoder])

    autoencoder.compile(optimizer="adam",loss=custom_loss)


    for epoch in trange(epochs):

        for step in range(steps):

            x = data[np.random.randint(0,len(data),batch_size)]
            autoencoder.train_on_batch(x,x)



    encoder.save(os.path.join("models","encoder_v{}.hdf5".format(__MODEL_VERSION__)))
    decoder.save(os.path.join("models","decoder_v{}.hdf5".format(__MODEL_VERSION__)))


