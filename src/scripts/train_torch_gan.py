import torch
import torch.optim as O

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.plots import  plot_multiple_images
from src.models.Critic import Critic
from src.models.Generator import Generator

import os
from src.config import *
from src.utils import initialize, create_or_cleanup

import matplotlib.pyplot as plt

GPU_DEVICE = torch.device(torch.cuda.current_device())

get_latent_variables = lambda batch_size=BATCH_SIZE: torch.rand((batch_size, LATENT_SIZE)).to(
        GPU_DEVICE) * 2 - 1

get_real_labels = lambda batch_size = BATCH_SIZE:1 - (torch.abs(torch.randn((batch_size,1)))*0.025).to(GPU_DEVICE)
get_fake_labels = lambda batch_size = BATCH_SIZE:-1 + (torch.abs(torch.randn((batch_size,1)))*0.025).to(GPU_DEVICE)

def calculate_gradient_penalty(critic,real_data,fake_data):

    rand_array = torch.rand((BATCH_SIZE,1)).to(GPU_DEVICE)
    interpolations = rand_array * real_data + (1-rand_array) * fake_data
    interpolations = interpolations.to(GPU_DEVICE)

    _,interpolation_results = critic(interpolations)

    gradients = torch.autograd.grad(
        outputs = interpolation_results,
        inputs = interpolations,
        grad_outputs=torch.ones(interpolation_results.size()).to(GPU_DEVICE),
        create_graph=True
    )[0]

    gradient_penalty = ((gradients.norm(2,dim=1) -1)**2) * LAMBDA

    return gradient_penalty




def main(
        data:torch.Tensor,
        matrix_dimension = MATRIX_DIMENSION,
        generator = None,
        critic = None,
        gradient_penalty = True
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


    x_train, x_test = train_test_split(data,  test_size=0.05)

    x_train = x_train.to(GPU_DEVICE)
    x_test = x_test.to(GPU_DEVICE)

    if generator == None:
        generator = Generator(matrix_dimension).to(GPU_DEVICE).apply(initialize)

    if critic == None:
        critic = Critic(matrix_dimension).to(GPU_DEVICE).apply(initialize)

    critic_optimizer = O.Adam(critic.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)
    generator_optimizer = O.Adam(generator.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)

    O.lr_scheduler.ReduceLROnPlateau(critic_optimizer,patience=2,factor=0.5)
    O.lr_scheduler.ReduceLROnPlateau(generator_optimizer,patience=2,factor=0.5)


    train_results = torch.zeros((EPOCH, 4))
    test_latent_variables = get_latent_variables(64)


    for epoch in trange(EPOCH):

        critic.train()
        generator.train()
        critic_optimizer.zero_grad()
        generator_optimizer.zero_grad()


        for step in range(STEPS_PER_EPOCH):

            for i in range(DISCRIMINATOR_STEP):

                critic.zero_grad()

                train_indices = torch.randint(0, len(x_train), (BATCH_SIZE,))
                train_batch = x_train[train_indices]

                _,real_loss = critic(train_batch)
                real_loss = real_loss.mean()

                z = get_latent_variables()
                fake_images = generator(z)

                _,fake_loss = critic(fake_images.detach())
                fake_loss = fake_loss.mean()
                wasserstein_loss = fake_loss - real_loss


                if gradient_penalty:
                    gp_loss = calculate_gradient_penalty(critic,train_batch,fake_images)
                    wasserstein_loss += gp_loss.mean()

                critic_optimizer.zero_grad()
                wasserstein_loss.backward()
                critic_optimizer.step()

                train_results[epoch, 0] += wasserstein_loss.item()


            generator_optimizer.zero_grad()
            z = get_latent_variables()

            fake_images = generator(z)
            _,generator_loss = critic(fake_images)
            generator_loss = -generator_loss.mean()
            generator_loss.backward()

            train_indices = torch.randint(0, len(x_train), (BATCH_SIZE,))
            train_batch = x_train[train_indices]
            real_features,_ = critic(train_batch)
            real_features = real_features.mean()

            z = get_latent_variables()
            fake_images = generator(z)
            fake_features, _ = critic(fake_images)
            fake_features = fake_features.mean()

            feature_loss = torch.abs(fake_features - real_features).mean()
            feature_loss.backward()

            train_results[epoch, 1] += generator_loss.item()
            train_results[epoch, 2] += feature_loss.item()
            generator_optimizer.step()

        generator.eval()
        results = generator(test_latent_variables).detach()

        torch.save(critic, os.path.join(MODELS_ROOT_DIR, "critic_last.pt"))
        torch.save(generator, os.path.join(MODELS_ROOT_DIR, "generator_last.pt"))

        if not epoch % CHECKPOINT_RATE:
            torch.save(critic,os.path.join(MODELS_ROOT_DIR,"critic_{}.pt".format(epoch)))
            torch.save(generator,os.path.join(MODELS_ROOT_DIR,"generator_{}.pt".format(epoch)))


        fig = plot_multiple_images(
            results.view(len(results), matrix_dimension, matrix_dimension).cpu().numpy()
            , 8)
        fig.savefig(
            os.path.join("..", "results", "training_{}".format(MODEL_VERSION),
                         "step_{}.png".format(STEPS_PER_EPOCH*(epoch+1))))
        plt.close(fig)

        test_indices = torch.randint(0, len(x_test), (TEST_BATCH, 1))
        test_batch = x_test[test_indices]

        critic.eval()
        _,test_output = critic(test_batch)
        real_test_loss = test_output.mean()
        _,fake_test_loss = critic(results)
        fake_test_loss = fake_test_loss.mean()

        train_results[epoch, 2] = real_test_loss.item()
        train_results[epoch, 3] = fake_test_loss.item()

        writer.add_scalar("Wasserstein Loss", train_results[epoch, 0] / STEPS_PER_EPOCH / DISCRIMINATOR_STEP,
                          epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)

        writer.add_scalar("Generator Loss", train_results[epoch, 1] / STEPS_PER_EPOCH, epoch * STEPS_PER_EPOCH)
        writer.add_scalar("Generator Feature Loss", train_results[epoch, 2], epoch * STEPS_PER_EPOCH )
        writer.add_scalar("Real Test Output", train_results[epoch, 3], epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("Fake Test Output", train_results[epoch, 4], epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("Critic Learning Rate",critic_optimizer.param_groups[0]["lr"],epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)
        writer.add_scalar("Generator Learning Rate",generator_optimizer.param_groups[0]["lr"],epoch * STEPS_PER_EPOCH)


    latent_variables = get_latent_variables(TEST_IMAGES)
    generator.eval()
    results = generator(latent_variables).detach()

    fig = plot_multiple_images(
        results.view(len(results), matrix_dimension, matrix_dimension).cpu().numpy(),
        8)
    fig.savefig(os.path.join("..", "results", "training_{}".format(MODEL_VERSION), "final.png"))
    plt.close(fig)


    writer.close()

