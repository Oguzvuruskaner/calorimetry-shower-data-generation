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
    
    interpolation_results = critic(interpolations)

    gradients = torch.autograd.grad(
        outputs = interpolation_results,
        inputs = interpolations,
        grad_outputs=torch.ones(interpolation_results.size()).to(GPU_DEVICE),
        create_graph=True
    )[0]

    gradient_penalty = ((gradients.norm(2,dim=1) -1)**2).mean() * LAMBDA

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

    critic_optimizer = O.Adam(critic.parameters(), lr=LEARNING_RATE)
    generator_optimizer = O.Adam(generator.parameters(), lr=LEARNING_RATE)


    train_results = torch.zeros((EPOCH, 3))
    test_latent_variables = get_latent_variables(80)

    for epoch in trange(EPOCH):

        critic.train()
        generator.train()

        for step in range(STEPS_PER_EPOCH):

            for i in range(DISCRIMINATOR_STEP):
                critic.zero_grad()

                train_indices = torch.randint(0, len(x_train), (BATCH_SIZE,))
                train_batch = x_train[train_indices]

                real_output = critic(train_batch).mean()
                real_loss = torch.abs(real_output-get_real_labels(1))
                real_loss.backward()

                z = get_latent_variables()
                fake_images = generator(z)

                fake_output = critic(fake_images.detach()).mean()
                fake_loss = torch.abs(fake_output-get_fake_labels(1))
                fake_loss.backward()

                wasserstein_loss = fake_output - real_output

                if gradient_penalty:
                    gp_loss = calculate_gradient_penalty(critic,train_batch,fake_images)
                    gp_loss.backward()
                    wasserstein_loss += gp_loss


                train_results[epoch, 0] += wasserstein_loss.item()

                critic_optimizer.step()


            z = get_latent_variables()

            fake_images = generator(z)
            fake_output = critic(fake_images)

            generator_loss = torch.abs(fake_output - get_real_labels()).mean()
            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            train_results[epoch, 1] += generator_loss.item()

        generator.eval()
        results = generator(test_latent_variables).detach()

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
        test_output = critic(test_batch)
        test_loss = test_output.mean()

        train_results[epoch, 2] = test_loss.item()

        writer.add_scalar("Wasserstein Loss", train_results[epoch, 0] / STEPS_PER_EPOCH / DISCRIMINATOR_STEP,
                          epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)

        writer.add_scalar("generator_loss", train_results[epoch, 1] / STEPS_PER_EPOCH, epoch * STEPS_PER_EPOCH)
        writer.add_scalar("test_loss", train_results[epoch, 2], epoch * STEPS_PER_EPOCH * DISCRIMINATOR_STEP)



    latent_variables = get_latent_variables(TEST_IMAGES)
    generator.eval()
    results = generator(latent_variables).detach()

    fig = plot_multiple_images(
        results.view(len(results), matrix_dimension, matrix_dimension).cpu().numpy(),
        8)
    fig.savefig(os.path.join("..", "results", "training_{}".format(MODEL_VERSION), "final.png"))
    plt.close(fig)



    writer.close()

