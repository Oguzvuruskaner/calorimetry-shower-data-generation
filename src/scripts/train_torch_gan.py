import torch
import torch.optim as O

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.plots import plot_data, plot_multiple_images
from src.models.Critic import Critic
from src.models.Generator import Generator

import os
from src.config import *
from src.utils import initialize, create_or_cleanup

import matplotlib.pyplot as plt



criterion = torch.nn.BCELoss(reduction="mean")


def main(
        data:torch.Tensor,
        labels: torch.Tensor,
        number_of_labels:int = NUMBER_OF_LABELS,
        matrix_dimension = MATRIX_DIMENSION,
        generator = None,
        critic = None
):

    for basename in os.listdir("train_logs"):
        os.unlink(os.path.join("train_logs",basename))

    RESULTS_ROOT_DIR = os.path.join("..","results","training_{}".format(MODEL_VERSION))
    MODELS_ROOT_DIR = os.path.join("..","models","training_{}".format(MODEL_VERSION))
    TRAIN_LOGS_DIR = os.path.join("train_logs")

    create_or_cleanup(RESULTS_ROOT_DIR)
    create_or_cleanup(MODELS_ROOT_DIR)
    create_or_cleanup(TRAIN_LOGS_DIR)

    writer = SummaryWriter(TRAIN_LOGS_DIR)

    gpu_device = torch.device(torch.cuda.current_device())

    # Taken by Rajarshee Mitra from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/10
    label_embedding = torch.nn.Embedding(number_of_labels, number_of_labels).to(gpu_device)
    label_embedding.weight.data = torch.eye(number_of_labels).to(gpu_device)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.05)

    x_train = x_train.to(gpu_device)
    y_train = y_train.to(gpu_device)
    x_test = x_test.to(gpu_device)
    y_test = y_test.to(gpu_device)

    if generator == None:
        generator = Generator(matrix_dimension, latent_size=LATENT_SIZE).to(gpu_device).apply(initialize)

    get_latent_variables = lambda batch_size=BATCH_SIZE: torch.randn((batch_size, LATENT_SIZE)).to(
        gpu_device) * 2 - 1

    get_real_labels = lambda batch_size=BATCH_SIZE: 1 - torch.abs(torch.randn((batch_size, 1)) * 0.05).to(
        gpu_device)
    get_fake_labels = lambda batch_size=BATCH_SIZE: torch.abs(torch.randn((batch_size, 1)) * 0.05).to(gpu_device)

    if critic == None:
        critic = Critic(matrix_dimension, number_of_labels).to(gpu_device).apply(initialize)

    critic_optimizer = O.Adam(critic.parameters(), lr=10e-5, weight_decay=1e-4)
    generator_optimizer = O.Adam(generator.parameters(), lr=10e-5, weight_decay=1e-4)

    train_results = torch.zeros((EPOCH, 7))
    test_train_results = torch.ones((TEST_BATCH, 1)).to(gpu_device)
    test_image_labels = torch.randint(0, 8, (80, 1)).to(gpu_device)
    test_latent_variables = get_latent_variables(80)

    for epoch in trange(EPOCH):

        for step in range(STEPS_PER_EPOCH):

            for i in range(DISCRIMINATOR_STEP):
                train_indices = torch.randint(0, len(x_train), (BATCH_SIZE,))
                train_batch = x_train[train_indices]
                label_batch = y_train[train_indices].view(BATCH_SIZE, 1)

                real_output, label_output = critic(train_batch)
                classification_loss = criterion(label_output,
                                                label_embedding(label_batch).view(-1, number_of_labels).detach())
                real_loss = criterion(real_output, get_real_labels())

                train_results[epoch, 0] += real_loss.item()
                train_results[epoch, 1] += classification_loss.item()

                z = get_latent_variables()
                labels = torch.randint(0, 9, (BATCH_SIZE, 1)).to(gpu_device)
                fake_images = generator(z, labels)

                fake_output, label_output = critic(fake_images.detach())
                fake_loss = criterion(fake_output, get_fake_labels())

                critic.zero_grad()
                (classification_loss + real_loss + fake_loss).backward()
                critic_optimizer.step()

                train_results[epoch, 2] += fake_loss.item()

            z = get_latent_variables()
            labels = torch.randint(0, 9, (BATCH_SIZE, 1)).to(gpu_device)

            fake_images = generator(z, labels)
            fake_output, label_output = critic(fake_images)

            generator_classification_loss = criterion(label_output,
                                                      label_embedding(labels).view(-1, number_of_labels).detach())
            generator_loss = criterion(fake_output, get_real_labels())

            generator.zero_grad()
            (generator_loss + generator_classification_loss).backward()
            generator_optimizer.step()

            train_results[epoch, 3] += generator_loss.item()
            train_results[epoch, 4] += generator_classification_loss.item()

        results = generator(test_latent_variables, test_image_labels).detach()

        if not epoch % CHECKPOINT_RATE:
            torch.save(critic,os.path.join(MODELS_ROOT_DIR,"critic_{}.pt".format(epoch)))
            torch.save(generator,os.path.join(MODELS_ROOT_DIR,"generator_{}.pt".format(epoch)))


        fig = plot_multiple_images(
            results.view(len(results), matrix_dimension, matrix_dimension).cpu().numpy()*HIT_E_MAX
            , 8)
        fig.savefig(
            os.path.join("..", "results", "training_{}".format(MODEL_VERSION),
                         "step_{}.png".format(STEPS_PER_EPOCH*(epoch+1))))
        plt.close(fig)

        test_indices = torch.randint(0, len(x_test), (TEST_BATCH, 1))
        test_batch = x_test[test_indices]
        test_labels = y_test[test_indices].view(TEST_BATCH, 1)

        test_output, test_class = critic(test_batch)
        test_loss = criterion(test_output, test_train_results)
        test_classification_loss = criterion(test_class,
                                             label_embedding(test_labels).view(-1, number_of_labels).detach())

        train_results[epoch, 5] = test_classification_loss.item()
        train_results[epoch, 6] = test_loss.item()

        writer.add_scalar("real_train_loss", train_results[epoch, 0] / STEPS_PER_EPOCH / DISCRIMINATOR_STEP,
                          epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("real_classification_loss",
                          train_results[epoch, 1] / STEPS_PER_EPOCH / DISCRIMINATOR_STEP,
                          epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("fake_train_loss", train_results[epoch, 2] / STEPS_PER_EPOCH / DISCRIMINATOR_STEP,
                          epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("generator_loss", train_results[epoch, 3] / STEPS_PER_EPOCH, epoch * STEPS_PER_EPOCH)
        writer.add_scalar("generator_classification_loss", train_results[epoch, 4] / STEPS_PER_EPOCH,
                          epoch * STEPS_PER_EPOCH)
        writer.add_scalar("test_classification_loss", train_results[epoch, 5],
                          epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("test_loss", train_results[epoch, 6], epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)



    latent_variables = get_latent_variables(TEST_IMAGES)
    results = generator(latent_variables).detach()

    fig = plot_multiple_images(
        results.view(len(results), matrix_dimension, matrix_dimension).cpu().numpy()*HIT_E_MAX,
        8)
    fig.savefig(os.path.join("..", "results", "training_{}".format(MODEL_VERSION), "final.png"))
    plt.close(fig)



    writer.close()

