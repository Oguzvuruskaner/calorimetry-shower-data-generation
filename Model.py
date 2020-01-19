from keras.models import Model,load_model
from keras.layers import Input,Dense,BatchNormalization,PReLU,Dropout
from keras.activations import tanh,sigmoid
from keras import backend as K
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from os.path import join


def wasserstein(y_true,y_pred):
    return -K.mean(y_pred*y_true)


def createCritic(input_size=10000,number_of_layers=5):

    input_layer = Input(shape=(input_size,))
    x = BatchNormalization()(input_layer)

    for i in range(number_of_layers):
        x = Dense(500)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(.3)(x)


    output_layer = Dense(1,activation=tanh)(x)

    model = Model(inputs=(input_layer,),outputs=output_layer)
    model.compile(optimizer="adam",loss=wasserstein)
    model.name = "Critic"

    return model


def createGenerator(input_size=1000,output_size=10000,number_of_layers=5):

    input_layer = Input(shape=(input_size,))
    x = BatchNormalization()(input_layer)

    for i in range(number_of_layers):
        x = Dense(500)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(.3)(x)

    output_layer = Dense(output_size,activation=sigmoid)(x)

    model = Model(inputs=(input_layer,),outputs=output_layer)
    model.compile(optimizer="adam",loss=wasserstein)
    model.name = "Generator"

    return model



def train(data,version,batch_size = 10000,epoch = 1000,critic_train_step=1000,clip_threshold = 0.01):

    data_norm = (data - np.min(data))/(np.max(data)-np.min(data))

    generator = createGenerator(output_size=batch_size)
    GAN_input = Input((generator.input_shape[1],))
    critic = createCritic(input_size=batch_size)
    critic.trainable = False

    GAN = Model(inputs=GAN_input,
                outputs=critic(generator(GAN_input)))

    GAN.compile(optimizer="adam",loss=wasserstein)

    print(GAN.summary())

    for epoch_number in range(epoch):

        print("Epoch {}".format(epoch_number+1))

        #Train critic with real data.
        rand_data = np.random.choice(data_norm,(critic_train_step,batch_size))
        y_values = np.ones((critic_train_step,1))
        critic.train_on_batch(rand_data,y_values)

        noise_input = np.random.normal(loc=0,scale=1,size=(critic_train_step,generator.input_shape[1]))
        pred = generator.predict(noise_input)

        #Train critic with fake data.
        y_values = -np.ones((critic_train_step, 1))
        critic.train_on_batch(pred, y_values)

        for layer in critic.layers:
            weights = layer.get_weights()
            weights = [np.clip(w,-clip_threshold,clip_threshold) for w in weights]
            layer.set_weights(weights)


        y_values = np.ones((critic_train_step, 1))
        noise_input = np.random.normal(loc=0, scale=1, size=(critic_train_step, generator.input_shape[1]))
        GAN.train_on_batch(noise_input, y_values)

        # Weight clipping

        critic.save(join("models","gen{}_critic.h5".format(version)))
        generator.save(join("models","gen{}_generator.h5".format(version)))

def validate(data,generator_path,number_of_batches=10,number_of_bins=40):

    #Load model from file.
    #wesserstein given in dictionary because it is not defined in keras.
    generator : Model = load_model(generator_path,{"wasserstein":wasserstein})
    data = (data - np.min(data)) / (np.max(data)-np.min(data))

    noise = np.random.normal(size=(number_of_batches,generator.input_shape[1]))
    pred = generator.predict(noise)
    pred.resize((generator.output_shape[1]*number_of_batches,))
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

    fig, axs = plt.subplots(1, 2,tight_layout=True)
    generated_histogram_plot = axs[0]
    data_histogram_plot = axs[1]

    generated_histogram_plot.hist(pred,bins=number_of_bins)
    generated_histogram_plot.yaxis.set_major_formatter(PercentFormatter(xmax=len(pred)))

    data_histogram_plot.hist(data,bins=number_of_bins)
    data_histogram_plot.yaxis.set_major_formatter(PercentFormatter(xmax=len(data)))

    plt.show()

def validate_critic(critic_path,generator_path,batch_size=100):

    critic : Model = load_model(critic_path,{
        "wasserstein":wasserstein
    })

    generator : Model = load_model(generator_path,{
        "wasserstein" : wasserstein
    })

    noise_array = np.random.normal(size=(batch_size,generator.input_shape[1]))
    pred_array = generator.predict(noise_array)

    result_array = critic.predict(pred_array)

    for result in result_array:
        print(result)

    return ...
