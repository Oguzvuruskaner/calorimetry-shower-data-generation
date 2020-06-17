from readRoot import create_jet_image_array
from scripts.problems import very_easy_problem, easy_problem,train_jet_generator
from scripts.scripts import create_npy_files, get_root_files

import os

if __name__ == "__main__":

    generator,critic,epoch_losses = very_easy_problem()
    generator.save(
        os.path.join("models", "very_easy_problem_generator.h5")
    )

    critic.save(
        os.path.join("models", "very_easy_problem_generator.h5")
    )


