from keras import Model
import seaborn as sns
import matplotlib.pyplot as plt
from scripts import createNpyFiles, loadAndSplitArray, filterOutliers, createScalers, plotFeatures, getSamples
from os.path import join
import numpy as np
from Model import train,loadModel

def main():
    ...
    createNpyFiles()
    filterOutliers()
    createScalers()
    plotFeatures()

def train_gan():

    loadAndSplitArray(join("npy","hit_e_combined.npy"),20)
    data = np.load(join("train_chunks","hit_e_combined_chunk_10.npy"),allow_pickle=True)
    data.resize(data.shape[0])

    train(data,1)


def evalulate_gan():

    GENERATE_SIZE = 10000

    generator : Model= loadModel(join("models","gen1_generator.h5"))


    input_noise = np.random.randn(GENERATE_SIZE,generator.input_shape[1])
    predicted_data = generator.predict(input_noise)
    predicted_data.resize((predicted_data.size,1))

    sns_plot = sns.distplot(predicted_data)

    figure = sns_plot.get_figure()
    figure.savefig(join("plots","generator_v1_plot"))


if __name__ == "__main__":

    # main()
    # train_gan()
    # evalulate_gan()
    # getSamples()
    # data = np.load(join("npy","hit_e_combined_1000000.npy"))
    # np.max(data)
    # sns.distplot(data,kde=False)
    # data = np.load(join("npy","hit_e_combined.npy"))
    # sns.distplot(data,kde=False)
    main()