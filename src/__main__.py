import os

from src.datasets.JetImageDataset import JetImageDataset
from src.training.models.autoencoders.JetImageCompressor import JetImageCompressor
from src.training.validation_tasks.JetImageComparison import JetImageComparison
from src.transformers.Flatten import Flatten
from src.transformers.Log10Scaling import Log10Scaling

if __name__ == "__main__":

    data = JetImageDataset()\
        .set_np_path(os.path.join("npy","jet_array_128x128.npy"))\
        .obtain(from_npy=True)\
        .add_invertible_transformation(Log10Scaling())\
        .add_pre_transformation(Flatten())\
        .apply_pre_transformations()\
        .array()

    model = JetImageCompressor(EPOCHS=1,STEPS=1)

    model.add_validation_task(
        JetImageComparison(
            os.path.join("results","autoencoder_results_v11")
        )
    )
    model.train(data)
    model.save()