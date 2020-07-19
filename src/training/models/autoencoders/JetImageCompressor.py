import os

import tensorflow as tf
import numpy as np

from tqdm import trange

from src.decorators.TrainingWrapper import TrainingWrapper
from src.training.TrainingModel import TrainingModel
import src.config as train_config


class JetImageCompressor(TrainingModel):
    BINARY_CROSSENTROPY = tf.keras.losses.BinaryCrossentropy()
    KL_DIVERGENCE = tf.keras.losses.KLDivergence()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.create_training_params(*args, **kwargs)
        self._models = self._create_models()
        self._save_dir = None
        self._load_dir = None

    @TrainingWrapper
    def train(self, data) -> "train_results":

        global EPOCHS, STEPS, BATCH_SIZE

        encoder, decoder = self._models
        autoencoder = tf.keras.Sequential([encoder, decoder])

        autoencoder.compile(loss=JetImageCompressor.custom_loss)
        self._models.append(autoencoder)

        train_results = np.zeros((EPOCHS, STEPS + 1))
        test_results = np.zeros((EPOCHS))

        for epoch in trange(EPOCHS):

            for step in range(STEPS):
                train_data = data[np.random.randint(0, len(data), (BATCH_SIZE))]
                train_results[epoch, step] = autoencoder.train_on_batch(train_data, train_data)

            test_data = data[np.random.randint(0, len(data), (BATCH_SIZE))]
            test_results[epoch] = autoencoder.test_on_batch(test_data,test_data)


        return (train_results, test_results)

    @BuilderMethod
    def set_save_dir(self, path: str):
        self._save_dir = path

    @BuilderMethod
    def set_load_dir(self, path: str):
        self._load_dir = path

    @BuilderMethod
    def save(self):

        global __MODEL_VERSION__

        encoder, decoder = self._models
        encoder.save(os.path.join(self._save_dir, "encoder_v{}.hdf5".format(__MODEL_VERSION__)))
        decoder.save(os.path.join(self._save_dir, "decoder_v{}.hdf5".format(__MODEL_VERSION__)))

    @BuilderMethod
    def load(self):

        global __MODEL_VERSION__

        self._models = [
            self._load_model(os.path.join(self._save_dir, "encoder_v{}.hdf5".format(__MODEL_VERSION__))),
            self._load_model(os.path.join(self._save_dir, "decoder_v{}.hdf5".format(__MODEL_VERSION__)))
        ]

    def _create_models(self):

        return [self._create_encoder(), self._create_decoder()]

    def _create_encoder(self):

        global DIMENSION, WHITE_NOISE_RATE, DROPOUT_RATE, N_COMPONENTS, __MODEL_VERSION__

        model = tf.keras.Sequential([

            tf.keras.Input(DIMENSION * DIMENSION),

            tf.keras.layers.Dropout(WHITE_NOISE_RATE),
            tf.keras.layers.Reshape((DIMENSION, DIMENSION, 1)),

            tf.keras.layers.Conv2D(32, (5, 5), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.MaxPool2D(),

            tf.keras.layers.Conv2D(64, (5, 5), padding="same"),
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

        ], name="encoder_v{}".format(__MODEL_VERSION__))

        model.summary()

        return model

    def _create_decoder(self):

        global DIMENSION, WHITE_NOISE_RATE, DROPOUT_RATE, N_COMPONENTS, __MODEL_VERSION__

        model = tf.keras.Sequential([
            tf.keras.Input(N_COMPONENTS),

            tf.keras.layers.Dense(DIMENSION * DIMENSION // 16),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(DROPOUT_RATE),

            tf.keras.layers.Reshape((DIMENSION // 4, DIMENSION // 4, 1)),

            tf.keras.layers.Conv2DTranspose(64, (5, 5), (2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(DROPOUT_RATE),

            tf.keras.layers.Conv2DTranspose(32, (5, 5), (2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(DROPOUT_RATE),

            tf.keras.layers.Conv2D(1, (5, 5), padding="same", activation="sigmoid"),

            tf.keras.layers.Flatten()

        ], name="decoder_v{}".format(__MODEL_VERSION__))

        model.summary()

        return model

    def create_training_params(self, *args, **kwargs):

        # Dynamically define variables

        self._training_params["N_COMPONENTS"] = kwargs.get("N_COMPONENTS", train_config.N_COMPONENTS)
        self._training_params["__MODEL_VERSION__"] = kwargs.get("__MODEL_VERSION__",
                                                             train_config.__MODEL_VERSION__)
        self._training_params["WHITE_NOISE_RATE"] = kwargs.get("WHITE_NOISE_RATE", 0.01)
        self._training_params["DROPOUT_RATE"] = kwargs.get("DROPOUT_RATE", 0.25)
        self._training_params["DIMENSION"] = kwargs.get("DIMENSION", train_config.DIMENSION)
        self._training_params["EPOCHS"] = kwargs.get("EPOCHS", 500)
        self._training_params["STEPS"] = kwargs.get("STEPS", 500)
        self._training_params["BATCH_SIZE"] = kwargs.get("BATCH_SIZE", 64)

        for key in self._training_params.keys():
            globals()[key] = self._training_params[key]

    @staticmethod
    def custom_loss(y_true, y_pred):
        return tf.keras.backend.mean(
            JetImageCompressor.BINARY_CROSSENTROPY(y_true, y_pred)
            + JetImageCompressor.KL_DIVERGENCE(y_true, y_pred)
        )

    def get_custom_objects(self):

        return {
            "custom_loss": JetImageCompressor.custom_loss
        }

    def _load_model(self, path):

        return tf.keras.models.load_model(path, self.get_custom_objects())
