from keras.models import Model,load_model
from keras.layers import Input, Dense, BatchNormalization, PReLU, Dropout, LeakyReLU, ReLU
from keras.activations import tanh
from keras import backend as K
import numpy as np
from os.path import join
from tqdm import tqdm


def wasserstein(y_true,y_pred):
    return -K.mean(y_pred*y_true)


def createCritic(input_size=1000,number_of_layers=3):

    input_layer = Input(shape=(input_size,))
    x = BatchNormalization()(input_layer)

    for i in range(number_of_layers):
        x = Dense(250)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(.3)(x)


    output_layer = Dense(1,activation=tanh)(x)

    model = Model(inputs=(input_layer,),outputs=output_layer)
    model.compile(optimizer="adam",loss=wasserstein)
    model.name = "Critic"

    return model


def createGenerator(input_size=100,output_size=1000,number_of_layers=3):

    input_layer = Input(shape=(input_size,))
    x = BatchNormalization()(input_layer)

    for i in range(number_of_layers):
        x = Dense(500)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(.3)(x)

    output_layer = Dense(output_size)(x)
    output_layer = ReLU(output_size)

    model = Model(inputs=(input_layer,),outputs=output_layer)
    model.compile(optimizer="adam",loss=wasserstein)
    model.name = "Generator"

    return model



def train(data,version=None,batch_size = 1000,epoch = 1000,critic_train_step=1000):

    data_norm = data
    # data_norm = (data - np.min(data))/(np.max(data)-np.min(data))

    generator = createGenerator(output_size=batch_size)
    GAN_input = Input((generator.input_shape[1],))
    critic = createCritic(input_size=batch_size)
    critic.trainable = False

    GAN = Model(inputs=GAN_input,
                outputs=critic(generator(GAN_input)))

    GAN.compile(optimizer="adam",loss=wasserstein)

    GAN.summary()

    for _ in tqdm(range(epoch)):


        #Train critic with real data.
        rand_data = np.random.choice(data_norm,(critic_train_step,batch_size))
        y_values = np.ones((critic_train_step,1))
        critic.train_on_batch(rand_data,y_values)

        noise_input = np.random.normal(loc=0,scale=1,size=(critic_train_step,generator.input_shape[1]))
        pred = generator.predict(noise_input)

        #Train critic with fake data.
        y_values = -np.ones((critic_train_step, 1))
        critic.train_on_batch(pred, y_values)

        # Weight clipping
        # for layer in critic.layers:
        #     weights = layer.get_weights()
        #     weights = [np.clip(w,-clip_threshold,clip_threshold) for w in weights]
        #     layer.set_weights(weights)


        y_values = np.ones((critic_train_step, 1))
        noise_input = np.random.normal(loc=0, scale=1, size=(critic_train_step, generator.input_shape[1]))
        GAN.train_on_batch(noise_input, y_values)

    critic.save(join("models","gen{}_critic.h5".format(version)))
    generator.save(join("models","gen{}_generator.h5".format(version)))


def loadModel(modelPath:str,verbose=True):
    model : Model = load_model(modelPath,{"wasserstein":wasserstein})
    if verbose:
        model.summary()
    return model

