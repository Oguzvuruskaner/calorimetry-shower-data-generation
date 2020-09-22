import torch
import os
from sklearn.cluster import KMeans

from src.scripts.train_torch_gan import main as train
import numpy as np


if __name__ == "__main__":

    JET_IMAGE_LABELS = 8

    jet_images = torch.load(os.path.join("..","data","jet_images.pt"))
    jet_images = jet_images.view(len(jet_images),-1)
    k_means = KMeans(JET_IMAGE_LABELS)
    labels = k_means.fit_transform(jet_images)
    #Getting argmin of smallest cluster center distance.
    labels = np.argmin(labels,axis=1)
    labels = torch.from_numpy(labels).view(len(labels),1)


    train(
        jet_images,
        labels,
        JET_IMAGE_LABELS+1,
    )