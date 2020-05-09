from Model import train_model, create_critic, create_generator
import numpy as np
from test_critic import train_critic,train_with_generator
from Model import wasserstein_loss
import os


def main():


    data = np.load(os.path.join("npy","triple_all_1000000.npy"))

    critic = create_critic()
    critic.compile(loss=wasserstein_loss,optimizer="adam" ,metrics=['accuracy'])

    generator = create_generator()

    train_with_generator(data,critic,generator)


if __name__ == "__main__":

    main()
