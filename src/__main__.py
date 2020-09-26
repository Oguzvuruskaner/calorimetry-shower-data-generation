import math

import torch
import os


from src.scripts.train_torch_gan import main as train, PLOT_ERROR

if __name__ == "__main__":

    PLOT_ERROR_LOG = -math.log(PLOT_ERROR)

    jet_images = torch.load(os.path.join("..","data","jet_images.pt"))
    jet_images = (torch.log(jet_images + PLOT_ERROR) + PLOT_ERROR_LOG) / (2*PLOT_ERROR_LOG)
    jet_images = jet_images.view(len(jet_images),-1)


    train(jet_images)