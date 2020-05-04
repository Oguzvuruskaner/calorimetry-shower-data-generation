from Model import loadModel
import os
import numpy as np
from tqdm import tqdm
from config import __MODEL_VERSION__
import seaborn as sns
import pandas as pd

def test_model(version = __MODEL_VERSION__):

    # Since all scripts are run on __main__
    # There is no need to give relative path to os.path.join
    generator = loadModel(os.path.join("models","gen{}_generator.h5".format(version)))
    critic = loadModel(os.path.join("models","gen{}_critic.h5".format(version)))


    test_for_generator(generator)
    test_for_critic(critic)


def test_for_critic(critic):
    ...

def test_for_generator(generator):
    # Plot generated values to the left and original values to the right.
    #

    df = pd.DataFrame()

    predictions_r = []
    predictions_z = []
    predictions_e = []

    for _ in tqdm(range(1000)):

        tmp_predictions = generator.predict(np.random.normal(0,1,(1,100))).reshape((3000,))
        predictions_r.append(tmp_predictions[::3])
        predictions_z.append(tmp_predictions[1::3])
        predictions_e.append(tmp_predictions[2::3])

    predictions_r = np.array(predictions_r)
    predictions_z = np.array(predictions_z)
    predictions_e = np.array(predictions_e)

    predictions_r.resize(predictions_r.size)
    predictions_z.resize(predictions_z.size)
    predictions_e.resize(predictions_e.size)

    df.assign(predictions_r=predictions_r)
    df.assign(predictions_z=predictions_z)
    df.assign(predictions_e=predictions_e)

    data = np.load(os.path.join("npy","{}_1000000.npy".format("hit_e")))
    df.assign(hit_e=data)
    data = np.load(os.path.join("npy","{}_1000000.npy".format("hit_r")))
    df.assign(hit_r=data)
    data = np.load(os.path.join("npy","{}_1000000.npy".format("hit_z")))
    df.assign(hit_z=data)

    
