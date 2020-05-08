from keras.models import Sequential,Model,load_model
import numpy as np
from os import unlink

MODEL_PATH = "weights_tmp.h5"

def train_critic(model:Model,data):

    model.save(MODEL_PATH)

    model.fit(data,verbose=0)

    model = load
    unlink(MODEL_PATH)