from keras.models import Sequential,Model,load_model
import numpy as np
from os import unlink,path
from keras.callbacks import TensorBoard,ReduceLROnPlateau
import datetime

import tensorflow as tf


MODEL_PATH = "weights_tmp.h5"



def train_critic(model:Model,data):
    # Model should have been compiled.

    # Code of the initiation of the TensorBoard callback
    # is taken from tensorflow.org/tensorboard/get_started

    reduce_lr_on_plateu = ReduceLROnPlateau(factor=.5,monitor="loss")

    y_values = np.ones((data.shape[0],1))

    model.fit(data,y_values,verbose=2,callbacks=[
        reduce_lr_on_plateu
    ],steps_per_epoch=500,epochs=200)

    unlink(MODEL_PATH)

def train_with_generator(data,critic,generator):

    data = data[::2]
    results =  generator.predict(np.random.normal(size=(data.shape[0],100)))
    x_values = np.concatenate((data,results))
    positive_label = np.ones((data.shape[0],1))
    negative_label = -np.ones((results.shape[0],1))
    y_values = np.append(positive_label,negative_label)

    critic.fit(x_values, y_values, verbose=1, steps_per_epoch=500, epochs=200)