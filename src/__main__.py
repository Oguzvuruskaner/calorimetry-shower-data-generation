from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.config import *
from src.helpers.JetDataset import JetDataset
from src.models.Critic import Critic
from src.models.Generator import Generator

import torch
import torch.optim as O
import os

from src.plots import plot_multiple_images
from src.utils import critic_init, generator_init, create_or_cleanup, decay_dropout_rate

import matplotlib.pyplot as plt

GPU_DEVICE = torch.device(torch.cuda.current_device())
get_latent_variables = lambda batch_size=BATCH_SIZE: torch.rand((batch_size, LATENT_SIZE)).to(
        GPU_DEVICE) * 2 - 1

LOAD_MODEL = True

if __name__ == "__main__":


    train = JetDataset(os.path.join("..","data","matrix_dataset","all.h5"))
    test = JetDataset(os.path.join("..","data","matrix_dataset","all_test.h5"))

    x_train_reader = lambda : iter(DataLoader(train,batch_size=BATCH_SIZE,shuffle=True))
    x_test_reader = lambda :iter(DataLoader(test,batch_size=TEST_BATCH,shuffle=True))

    generator = Generator(MATRIX_DIMENSION).apply(generator_init).to(GPU_DEVICE)
    critic = Critic(MATRIX_DIMENSION).apply(critic_init).to(GPU_DEVICE)
    for basename in os.listdir("train_logs"):
        os.unlink(os.path.join("train_logs",basename))

    RESULTS_ROOT_DIR = os.path.join("..","results","training_{}".format(MODEL_VERSION))
    MODELS_ROOT_DIR = os.path.join("..","models","training_{}".format(MODEL_VERSION))
    TRAIN_LOGS_DIR = os.path.join("train_logs")

    create_or_cleanup(RESULTS_ROOT_DIR)
    create_or_cleanup(MODELS_ROOT_DIR)
    create_or_cleanup(TRAIN_LOGS_DIR)

    writer = SummaryWriter(TRAIN_LOGS_DIR)

    train_results = torch.zeros((EPOCH, 7))

    critic_optimizer = O.Adam(critic.parameters(), lr=LEARNING_RATE*20)
    generator_optimizer = O.Adam(generator.parameters(), lr=LEARNING_RATE)

    critic_optimizer_scheduler = O.lr_scheduler.ExponentialLR(critic_optimizer,gamma=.95)
    generator_optimizer_scheduler = O.lr_scheduler.ExponentialLR(generator_optimizer, gamma=.97)

    reference_variables = get_latent_variables(TEST_IMAGES)



    for epoch in trange(EPOCH):

        critic.train()
        generator.train()
        critic_optimizer.zero_grad()
        generator_optimizer.zero_grad()


        for step in range(STEPS_PER_EPOCH):

            for i in range(DISCRIMINATOR_STEP):

                critic.zero_grad()

                train_batch,_ = next(x_train_reader())

                real_output = critic(train_batch.to(GPU_DEVICE)).mean()
                train_results[epoch, 0] += real_output.item()


                z = get_latent_variables()
                fake_images = generator(z)
                fake_output = critic(fake_images.detach()).mean()
                train_results[epoch, 1] += fake_output.item()

                wasserstein_loss = fake_output - real_output

                critic_optimizer.zero_grad()

                wasserstein_loss.backward()
                critic_optimizer.step()


            generator_optimizer.zero_grad()
            z = get_latent_variables()

            fake_images = generator(z)
            generator_loss = -critic(fake_images).mean()
            generator_loss.backward()

            train_batch,_ = next(x_train_reader())
            real_features = critic(train_batch.to(GPU_DEVICE),True).mean()

            z = get_latent_variables()
            fake_images = generator(z)
            fake_features = critic(fake_images,True).mean()

            feature_loss = torch.abs(fake_features - real_features).mean()
            feature_loss.backward()

            train_results[epoch, 2] += generator_loss.item()
            train_results[epoch, 3] += feature_loss.item()
            generator_optimizer.step()



        generator.eval()
        results = generator(reference_variables).detach()

        torch.save(critic, os.path.join(MODELS_ROOT_DIR, "critic_last.pt"))
        torch.save(generator, os.path.join(MODELS_ROOT_DIR, "generator_last.pt"))

        if not epoch % CHECKPOINT_RATE:
            torch.save(critic,os.path.join(MODELS_ROOT_DIR,"critic_{}.pt".format(epoch)))
            torch.save(generator,os.path.join(MODELS_ROOT_DIR,"generator_{}.pt".format(epoch)))



        fig = plot_multiple_images(
            results.view(len(results), MATRIX_DIMENSION, MATRIX_DIMENSION).cpu().numpy()
            , 8)
        fig.savefig(
            os.path.join("..", "results", "training_{}".format(MODEL_VERSION),
                         "step_{}.png".format(STEPS_PER_EPOCH*(epoch+1))))
        plt.close(fig)

        test_batch,_ = next(x_test_reader())

        critic.eval()
        real_test_result = critic(test_batch.to(GPU_DEVICE)).mean()
        fake_test_result = critic(results).mean()

        train_results[epoch, 4] = real_test_result.item()
        train_results[epoch, 5] = fake_test_result.item()

        writer.add_scalar("Real Train Output", train_results[epoch, 0] / STEPS_PER_EPOCH,DISCRIMINATOR_STEP * epoch * STEPS_PER_EPOCH)
        writer.add_scalar("Fake Train Output", train_results[epoch, 1] / STEPS_PER_EPOCH,DISCRIMINATOR_STEP * epoch * STEPS_PER_EPOCH)
        writer.add_scalar("Generator Loss", train_results[epoch, 2] / STEPS_PER_EPOCH, epoch * STEPS_PER_EPOCH)
        writer.add_scalar("Generator Feature Loss", train_results[epoch, 3], epoch * STEPS_PER_EPOCH )
        writer.add_scalar("Real Test Output", train_results[epoch, 4], epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("Fake Test Output", train_results[epoch, 5], epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("Critic Learning Rate",critic_optimizer.param_groups[0]["lr"],epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("Generator Learning Rate",generator_optimizer.param_groups[0]["lr"],epoch * STEPS_PER_EPOCH)

        critic_optimizer_scheduler.step()
        generator_optimizer_scheduler.step()
        decay_dropout_rate(critic)
        decay_dropout_rate(generator)

    latent_variables = get_latent_variables(TEST_IMAGES)
    generator.eval()
    results = generator(latent_variables).detach()

    fig = plot_multiple_images(
        results.view(len(results), MATRIX_DIMENSION, MATRIX_DIMENSION).cpu().numpy(),
        8)
    fig.savefig(os.path.join("..", "results", "training_{}".format(MODEL_VERSION), "final.png"))
    plt.close(fig)


    writer.close()
