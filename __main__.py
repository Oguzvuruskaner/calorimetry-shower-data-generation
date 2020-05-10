from Model import train_model, create_critic, create_generator
import numpy as np

from scripts.samples import get_samples
from test_critic import train_critic,train_with_generator
from Model import wasserstein_loss
import os

from scripts.scripts import create_npy_files

def main():



    data = np.load(os.path.join("npy","triple_all_1000000.npy"))
    train_model(data,1)

if __name__ == "__main__":

    main()
