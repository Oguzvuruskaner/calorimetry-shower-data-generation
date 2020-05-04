from Model import train_model, plot_data
import numpy as np
import os
from config import __MODEL_VERSION__
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scripts.test_model import test_model

def plot_all_data():

    data = np.load(os.path.join("npy","triple_all.npy"))
    plot_data(data[:,0],"All R Data",os.path.join("plots","all_r_data.png"))
    plot_data(data[:,1],"All Z Data",os.path.join("plots","all_z_data.png"))
    plot_data(data[:,2],"All E Data",os.path.join("plots","all_e_data.png"))


def main():

    data = np.load(os.path.join("npy","triple_all_1000000.npy"))
    train_model(data)


if __name__ == "__main__":

    main()
