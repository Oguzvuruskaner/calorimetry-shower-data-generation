from keras.models import Model,load_model
from keras.layers import Input, Dense, BatchNormalization, Flatten, LeakyReLU, LocallyConnected1D, \
    Reshape, Dropout, concatenate
from keras import backend as K, Sequential
import numpy as np
from os.path import join
from keras.optimizers import Adam
from keras.utils import plot_model
from tqdm import tqdm
from config import __MODEL_VERSION__
from scripts.test_model import test_critic, generate_fake_data, plot_loss


def wasserstein_loss(y_true, y_pred):
    return K.mean(-y_true * y_pred)


# As it is decided from the last conference,
# the genreator is going to try to generate the
# hypothesis that hit_r,hit_e and hit_z values are only dependent to each other.

def create_generator(input_size = 100,version=__MODEL_VERSION__):

    # Since z values may be negative,
    # Relu should be used only for r and e values.
    model = Sequential([
        Dense(256,input_dim=input_size),
        LeakyReLU(),
    ],name="generator_{}".format(version))

    # Adding hidden layers
    for i in range(4):
        model.add(Dense(256))
        model.add(Dropout(.3))
        model.add(LeakyReLU())
        model.add(BatchNormalization())

    model.add(Dense(3))


    return model


def create_critic(input_size=3,version=__MODEL_VERSION__):


    model = Sequential([
        Dense(10,input_dim=input_size),
        LeakyReLU(),
    ],name="critic_{}".format(version))

    # Adding hidden layers
    for i in range(4):
        model.add(Dense(10))
        model.add(Dropout(.3))
        model.add(LeakyReLU())
        model.add(BatchNormalization())

    model.add(Dense(1, activation="tanh"))


    return model

def create_gan(generator:Model, critic:Model,noise_size=100):

    generator_input = Input(shape=(noise_size,))

    full_generator = generator(generator_input)

    for layer in generator.layers:
        layer.trainable = True
    generator.trainable = True

    for layer in critic.layers:
        layer.trainable = False
    critic.trainable = False


    critic_generator_combined = critic(full_generator)

    GAN = Model(inputs=[generator_input],outputs=[critic_generator_combined])

    GAN.compile(optimizer=Adam(0.0001,beta_1=0.5,beta_2=0.9),loss=wasserstein_loss)
    GAN.summary()

    return GAN

def create_full_discriminator_model(critic,generator,noise_size=100):

    for layer in critic.layers:
        layer.trainable = True
    critic.trainable = True

    for layer in generator.layers:
        layer.trainable = False
    generator.trainable = False

    fake_data_input = Input(shape=(noise_size,))
    generator_with_input = generator(fake_data_input)
    critic_with_fake_data = critic(generator_with_input)

    real_data_input = Input(shape=(3,))
    critic_with_real_data = critic(real_data_input)

    full_discriminator = Model(inputs=[real_data_input,fake_data_input],
                               outputs=[critic_with_real_data,critic_with_fake_data])

    full_discriminator.compile(optimizer=Adam(0.0001,beta_1=0.5,beta_2=0.5),loss=[
        wasserstein_loss,
        wasserstein_loss
    ])
    full_discriminator.summary()
    return full_discriminator


def train_model(data,version = __MODEL_VERSION__,epochs = 50,steps_per_epoch=100,batch_size=5):

    generator = create_generator()

    critic = create_critic()
    gan = create_gan(generator,critic)
    full_discriminator = create_full_discriminator_model(critic,generator)

    discriminator_loss = []
    generator_loss = []


    for _ in tqdm(range(epochs)):

        discriminator_tmp = []
        generator_tmp = []

        for _ in range(steps_per_epoch):

            indices_array = np.random.choice(np.arange(len(data)),batch_size)

            true_input = data[indices_array]
            fake_input = np.random.normal(0,1,(batch_size,100))

            true_label = np.ones((batch_size,1))
            fake_label = - true_label

            train_result = full_discriminator.train_on_batch([true_input,fake_input],[true_label,fake_label])
            discriminator_tmp.append(np.mean(train_result))

            train_result = gan.train_on_batch([fake_input],[true_label])
            generator_tmp.append(train_result)

        generator_loss.append(np.mean(generator_tmp))
        discriminator_loss.append(np.mean(discriminator_tmp))


    generator.save(join("models","gen{}_generator.h5".format(version)))
    critic.save(join("models","gen{}_critic.h5".format(version)))

    plot_model(generator, join("models", "gen{}_generator_model.png".format(version))  ,show_shapes=True)
    plot_model(critic, join("models", "gen{}_critic_model.png".format(version)),show_shapes=True)

    plot_loss(generator_loss,epochs,"gen{}_generator_loss.png".format(version),"gen{}_generator Loss".format(version))
    plot_loss(generator_loss,epochs,"gen{}_critic_loss.png".format(version),"gen{}_critic Loss".format(version))
    generate_fake_data(generator,version=version)
    test_critic(data,critic,version=version)





def loadModel(model_path:str):

    # For additional variables in the model,
    # the developer should declare a dictionary

    variable_dict = {}
    model = load_model(model_path,variable_dict)
    return model

