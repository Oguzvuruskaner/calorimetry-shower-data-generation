import pickle

from keras.initializers import RandomNormal
from keras.models import Model,load_model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU,Dropout, concatenate
from keras import backend as K, Sequential
import numpy as np
import os
from keras.optimizers import RMSprop
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from config import __MODEL_VERSION__
from scripts.test_model import test_critic, generate_fake_data, plot_loss
from keras.constraints import Constraint
from keras.activations import sigmoid

class ClipConstraint(Constraint):

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


KERNEL_CONSTRAINT = ClipConstraint(clip_value=0.02)
KERNEL_INITIALIZER = RandomNormal()


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# As it is decided from the last conference,
# the genreator is going to try to generate the
# hypothesis that hit_r,hit_e and hit_z values are only dependent to each other.

def create_generator(input_size = 100,version=__MODEL_VERSION__):

    # Since z values may be negative,
    # Relu should be used only for r and e values.


    model = Sequential([
        Dense(128,input_dim=input_size,kernel_initializer = KERNEL_INITIALIZER,kernel_constraint=KERNEL_CONSTRAINT),
        LeakyReLU(),
    ],name="generator_{}".format(version))

    # Adding hidden layers
    for i in range(10):

        model.add(Dense(128,kernel_initializer = KERNEL_INITIALIZER,kernel_constraint=KERNEL_CONSTRAINT))
        model.add(Dropout(.3))
        model.add(LeakyReLU())
        model.add(BatchNormalization())

    model.add(Dense(3,activation=sigmoid))


    return model


def create_critic(input_size=3,version=__MODEL_VERSION__):


    model = Sequential([
        Dense(128,input_dim=input_size,kernel_initializer = KERNEL_INITIALIZER,kernel_constraint=KERNEL_CONSTRAINT),
        LeakyReLU(),
    ],name="critic_{}".format(version))

    # Adding hidden layers
    for i in range(6):
        model.add(Dense(128,kernel_initializer = KERNEL_INITIALIZER,kernel_constraint=KERNEL_CONSTRAINT))
        model.add(Dropout(.3))
        model.add(LeakyReLU())
        model.add(BatchNormalization())

    model.add(Dense(1,activation="tanh"))


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

    GAN.compile(optimizer=RMSprop(learning_rate=0.00005),loss=wasserstein_loss)
    GAN.summary()

    return GAN

def create_full_discriminator_model(critic,generator,noise_size=100):

    for layer in critic.layers:
        layer.trainable = True
    critic.trainable = True

    for layer in generator.layers:
        layer.trainable = False
    generator.trainable = False

    noise_input = Input(shape=(noise_size,))
    generator_with_input = generator(noise_input)
    critic_with_fake_data = critic(generator_with_input)

    real_data_input = Input(shape=(3,))
    critic_with_real_data = critic(real_data_input)

    full_discriminator = Model(inputs=[real_data_input,noise_input],
                               outputs=[critic_with_real_data,critic_with_fake_data])

    full_discriminator.compile(optimizer=RMSprop(0.00005),loss=[
        wasserstein_loss,
        wasserstein_loss
    ])
    full_discriminator.summary()
    return full_discriminator



def train_model(data,version = __MODEL_VERSION__,epochs = 200,steps_per_epoch=500,mini_batch_size=25):
    r_scaler = MinMaxScaler()
    z_scaler = MinMaxScaler()
    e_scaler = MinMaxScaler()

    r_scaler.fit(data[:,0].reshape(-1,1))
    z_scaler.fit(data[:,1].reshape(-1,1))
    e_scaler.fit(data[:,2].reshape(-1,1))

    pickle.dump(r_scaler,open("r_scaler.pkl","wb"))
    pickle.dump(z_scaler,open("z_scaler.pkl","wb"))
    pickle.dump(e_scaler,open("e_scaler.pkl","wb"))

    data[:,0] = r_scaler.transform(data[:,0].reshape(-1,1)).reshape((-1,))
    data[:,1] = z_scaler.transform(data[:,1].reshape(-1,1)).reshape((-1,))
    data[:,2] = e_scaler.transform(data[:,2].reshape(-1,1)).reshape((-1,))

    generator = create_generator()

    critic = create_critic()
    gan = create_gan(generator,critic)
    full_discriminator = create_full_discriminator_model(critic,generator)

    loss_data = np.zeros((epochs,3))
    tmp_data = np.zeros((steps_per_epoch,3))


    for epoch in tqdm(range(epochs)):

        for step in range(steps_per_epoch):

            indices_array = np.random.choice(np.arange(len(data)),mini_batch_size)

            true_input = data[indices_array]
            fake_input = np.random.normal(0,1,(mini_batch_size,100))

            true_label = -np.ones((mini_batch_size,1))
            fake_label = - true_label

            critic_real_loss,critic_fake_loss,_ = full_discriminator.train_on_batch([true_input,fake_input],[true_label,fake_label])


            fake_input = np.random.normal(0,1,(mini_batch_size//5,100))
            true_label = -np.ones((mini_batch_size//5,1))

            generated_loss = gan.train_on_batch([fake_input],[true_label])

            tmp_data[step,0] = critic_real_loss
            tmp_data[step,1] = critic_fake_loss
            tmp_data[step,2] = generated_loss

        loss_data[epoch,:] = np.mean(tmp_data.T,axis=1)

    plot_loss(loss_data,os.path.join("results","v_{}_loss.png".format(version)))

    generator.save(os.path.join("models","gen{}_generator.h5".format(version)))
    critic.save(os.path.join("models","gen{}_critic.h5".format(version)))

    plot_model(generator, os.path.join("models", "gen{}_generator_model.png".format(version))  ,show_shapes=True)
    plot_model(critic, os.path.join("models", "gen{}_critic_model.png".format(version)),show_shapes=True)

    generate_fake_data(generator,generated_experiments=50,version=version)
    test_critic(data[::100],critic,version=version)


