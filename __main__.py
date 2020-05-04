from Model import train_model
import numpy as np
import os
from config import __MODEL_VERSION__
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scripts.test_model import test_model


def main():

    data = np.load(os.path.join("npy","triple_all_1000000.npy"))
    train_model(data,version=__MODEL_VERSION__)

if __name__ == "__main__":

    main()
