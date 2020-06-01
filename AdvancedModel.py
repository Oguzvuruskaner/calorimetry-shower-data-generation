import tensorflow as tf
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow_core.python.keras import Model
from tqdm import tqdm

from Model import wasserstein_loss
from config import __MODEL_VERSION__,HIT_R_MAX,HIT_R_MIN,HIT_Z_MAX,HIT_Z_MIN,DIMENSION


NOISE_INPUT_SIZE = 100
SLICE = 1000

np.random.seed(12)



class JetDataset(tf.data.Dataset):

    def __new__(cls,num_samples):


        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(num_samples)
        )

    def _generator(self,num_samples):

        data = np.load(os.path.join("npy","per_jet_all.npy"),allow_pickle=True)
        total_samples = len(data)

        get_random_number = lambda  : np.random.randint(0,total_samples,(num_samples,1))

        while True:

            random_indices = get_random_number()
            yield(data[random_indices])


def convert_to_2d_hist(jet:np.array,jet_size:int) -> np.ndarray:

    # First n elements are jet_size.

    ret_array = np.zeros((DIMENSION,DIMENSION,1))
    tmp_jet = jet[:jet_size]
    tmp_jet[:,0] = np.floor((tmp_jet[:,0] - HIT_R_MIN)/ (HIT_R_MAX - HIT_R_MIN) *DIMENSION)
    tmp_jet[:,1] = np.floor((tmp_jet[:,1] - HIT_Z_MIN)/ (HIT_Z_MAX - HIT_Z_MIN) *DIMENSION)

    for hit_r,hit_z,hit_e in jet:
        ret_array[int(hit_z,), int(hit_r)] += hit_e


    return ret_array

def subtract_from_image(jet_image:np.array,particle) -> np.array:

    ret_array = np.copy(jet_image)

    particle[0] = np.floor((particle[0] - HIT_R_MIN) / (HIT_R_MAX - HIT_R_MIN) * DIMENSION)
    particle[1] = np.floor((particle[1] - HIT_Z_MIN) / (HIT_Z_MAX - HIT_Z_MIN) * DIMENSION)

    ret_array[int(particle[0] ), int(particle[1])] -= particle[2]

    return ret_array


# Finish check is for checking the generation.

def create_generator(noise_input_size:int = NOISE_INPUT_SIZE) -> (Model,Model):


    image_input = tf.keras.layers.Input(shape=(DIMENSION*DIMENSION,))
    image_layer = tf.keras.layers.Reshape(target_shape=(DIMENSION,DIMENSION,1))(image_input)

    for i in range(3):

        image_layer = tf.keras.layers.ZeroPadding2D(padding=1)(image_layer)
        image_layer = tf.keras.layers.LocallyConnected2D(16,(3,3))(image_layer)
        image_layer = tf.keras.layers.MaxPooling2D()(image_layer)
        image_layer = tf.keras.layers.LayerNormalization()(image_layer)

    image_layer = tf.keras.layers.Flatten()(image_layer)
    image_layer = tf.keras.layers.Dense(DIMENSION*DIMENSION)(image_layer)
    image_layer = tf.keras.layers.Dropout(rate=0.5)(image_layer)
    image_layer = tf.keras.layers.LeakyReLU()(image_layer)
    image_layer = tf.keras.layers.BatchNormalization()(image_layer)

    noise_input = tf.keras.layers.Input(shape=(noise_input_size,))
    noise_layer = tf.keras.layers.Dense(DIMENSION*DIMENSION)(noise_input)
    noise_layer = tf.keras.layers.Dropout(rate=0.5)(noise_layer)
    noise_layer = tf.keras.layers.LeakyReLU()(noise_layer)
    noise_layer = tf.keras.layers.BatchNormalization()(noise_layer)

    merged_layers = tf.keras.layers.concatenate([image_layer,noise_layer])

    output = tf.keras.layers.Dense(3,activation = "sigmoid")(merged_layers)

    finished_generate = tf.keras.layers.Dense(1,activation="sigmoid")(image_layer)


    generator_model = tf.keras.Model(
        inputs=[noise_input,image_input],
        outputs=[output],
        name="generator_{}".format(__MODEL_VERSION__)
    )
    generator_model.summary()

    for layer in generator_model.layers:
        layer.trainable = False

    finished_check_model = tf.keras.Model(
        inputs=[image_input],
        outputs=[finished_generate]
    )

    finished_check_model.compile(loss="binary_crossentropy")

    for layer in generator_model.layers:
        layer.trainable = True

    return (generator_model,finished_check_model)


def create_critic():

    image_input = tf.keras.layers.Input(shape=(DIMENSION * DIMENSION,))
    image_layer = tf.keras.layers.Reshape(target_shape=(DIMENSION, DIMENSION, 1))(image_input)

    for i in range(3):
        image_layer = tf.keras.layers.ZeroPadding2D(padding=1)(image_layer)
        image_layer = tf.keras.layers.LocallyConnected2D(16, (3, 3))(image_layer)
        image_layer = tf.keras.layers.MaxPooling2D()(image_layer)
        image_layer = tf.keras.layers.LayerNormalization()(image_layer)


    image_layer = tf.keras.layers.Flatten()(image_layer)
    image_layer = tf.keras.layers.Dense(DIMENSION * DIMENSION)(image_layer)
    image_layer = tf.keras.layers.Dropout(rate=0.5)(image_layer)
    image_layer = tf.keras.layers.LeakyReLU()(image_layer)
    image_layer = tf.keras.layers.BatchNormalization()(image_layer)

    noise_input = tf.keras.layers.Input(shape=(3,))
    noise_layer = tf.keras.layers.Dense(DIMENSION * DIMENSION)(noise_input)
    noise_layer = tf.keras.layers.Dropout(rate=0.5)(noise_layer)
    noise_layer = tf.keras.layers.LeakyReLU()(noise_layer)
    noise_layer = tf.keras.layers.BatchNormalization()(noise_layer)

    merged_layers = tf.keras.layers.concatenate([image_layer, noise_layer])

    output = tf.keras.layers.Dense(1, activation="tanh")(merged_layers)

    model = tf.keras.Model(
        inputs=[noise_input, image_input],
        outputs=[output],
        name="critic_{}".format(__MODEL_VERSION__)

    )

    model.summary()

    return model


def create_gan(critic:Model,generator:Model):

    image_layer = tf.keras.layers.Input(shape=(DIMENSION*DIMENSION,))
    noise_layer = tf.keras.layers.Input(shape=(NOISE_INPUT_SIZE,))

    critic.trainable = False
    for layer in critic.layers:
        layer.trainable = False

    generator.trainable = True
    for layer in generator.layers:
        layer.trainable = True


    GAN = Model(
        inputs=[noise_layer,image_layer],
        outputs=[critic(
            [generator(
                [noise_layer,image_layer]
            ),image_layer])],
        name="GAN_{}".format(__MODEL_VERSION__)
    )

    GAN.compile("rmsprop",wasserstein_loss)
    GAN.summary()

    return GAN




def train_model(data,epochs = 200,steps_per_epoch = 500,mini_batch_size=5):


    critic = create_critic()
    generator,finish_check = create_generator()

    GAN = create_gan(critic,generator)

    critic.trainable = True
    for layer in critic.layers:
        layer.trainable = True

    critic.compile("rmsprop",wasserstein_loss)

    for epoch in tqdm(range(epochs)):

        for step in range(steps_per_epoch):

            critic_data = data[np.random.randint(0,len(data),(mini_batch_size,1))]

            for jet in critic_data:

                # images holds transitions
                # For example
                # get_image converts jet to jet image.
                # images[0] = get_image(jet)
                # images[1] = get_image( jet - jet[-1])
                # images[2] = get_image( jet - jet[-1] - jet[-2])
                # Goes on and on...
                jet = jet[0]
                images = np.zeros((len(jet), DIMENSION, DIMENSION, 1))

                images[0] = np.histogram2d(x = jet[:,0],y = jet[:,1],bins=DIMENSION,weights=jet[:,2])[0].reshape((DIMENSION,DIMENSION,1))

                for ind,particle in enumerate(jet[:-1],start=1):

                    images[ind] = subtract_from_image(images[ind-1],particle)

                images.resize((len(jet),DIMENSION*DIMENSION))
                # jet is reversed because images array is filled this way.

                positive_labels = np.ones((len(jet),1))

                for i in tqdm(range(SLICE)):

                    critic.train_on_batch([jet[i::SLICE],images[i::SLICE]],positive_labels[i::SLICE])






