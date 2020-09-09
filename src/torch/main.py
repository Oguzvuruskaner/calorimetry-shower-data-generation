import torchvision
import torch
import torch.optim as O

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.torch.Critic import Critic
from src.torch.Encoder import Encoder
from src.torch.Generator import Generator

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os
from random import choice

from src.torch.utils import initialize

ROOT_PATH = os.path.join("mnist_data")

EPOCH = 400
STEPS_PER_EPOCH = 50
BATCH_SIZE = 256

DIMENSION = 28

criterion = torch.nn.BCELoss()


if __name__ == "__main__":


    writer = SummaryWriter("mnist_logs")

    gpu_device = torch.device(torch.cuda.current_device())

    mnist = MNIST(
        ROOT_PATH,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )


    mnist.data = torch.div(mnist.data, 255.)
    data = mnist.data
    data = data.view(len(data), 1, DIMENSION, DIMENSION)
    train_data, test_data = train_test_split(data)


    train_data = train_data.to(gpu_device, torch.float32)
    test_data = test_data.to(gpu_device, torch.float32)

    generator = Generator(28).to(gpu_device, torch.float32).apply(initialize)

    latent_variables = torch.randn((64, 100)).to(gpu_device)

    real_label = torch.FloatTensor([1]).to(gpu_device)
    fake_label = torch.FloatTensor([0]).to(gpu_device)


    critic = Critic(28).to(gpu_device, torch.float32).apply(initialize)

    critic_optimizer = O.Adam(critic.parameters(),lr=0.1)
    generator_optimizer = O.Adam(generator.parameters(),lr=0.1)

    train_results = torch.zeros((EPOCH, 4))

    k_t = 0
    gamma = 0.001
    k_t_lr = 0.1

    for epoch in trange(EPOCH):

        generator_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        for step in range(STEPS_PER_EPOCH):

            for i in range(5):

                critic.zero_grad()

                train_indices = torch.randint(0, len(train_data), (BATCH_SIZE,))
                train_batch = train_data[train_indices]

                real_output = critic(train_batch)
                real_output = real_output.mean(0).view(1)
                real_loss = criterion(real_output,real_label)
                train_results[epoch, 0] += real_loss.item()

                z = torch.randn(BATCH_SIZE, 100) \
                    .to(gpu_device)

                fake_images = generator(z)

                fake_output = critic(fake_images.detach())
                fake_output = fake_output.mean(0).view(1)
                fake_loss = criterion(fake_output, fake_label)

                critic_loss = real_loss - k_t * fake_loss


                train_results[epoch, 1] += fake_loss.item()

                critic_optimizer.step()

            generator.zero_grad()

            z = torch.randn(BATCH_SIZE, 100) \
                .to(gpu_device)

            fake_images = generator(z)

            fake_output = critic(fake_images)
            fake_output = fake_output.mean(0)

            loss = -criterion(fake_output,fake_label)
            loss.backward()

            train_results[epoch, 2] += loss.item()

            generator_optimizer.step()
            k_t += k_t_lr * (gamma * real_loss.item() - fake_loss.item())



        writer.add_scalar("real_train_loss",train_results[epoch,0]/STEPS_PER_EPOCH/5,epoch*STEPS_PER_EPOCH)
        writer.add_scalar("fake_train_loss",train_results[epoch,1]/STEPS_PER_EPOCH/5,epoch*STEPS_PER_EPOCH)
        writer.add_scalar("generator_middlelayer_loss",train_results[epoch,2]/STEPS_PER_EPOCH,epoch*STEPS_PER_EPOCH)



    latent_variables = torch.randn((BATCH_SIZE,100)).to(gpu_device)
    results = generator(latent_variables)
    results_image = torchvision.utils.make_grid(results)

    torchvision.utils.save_image(results_image,"results.png")


    with open("generator.pkl", "wb") as fp:
        torch.save(generator, fp)

    with open("critic.pkl", "wb") as fp:
        torch.save(critic, fp)


    writer.close()