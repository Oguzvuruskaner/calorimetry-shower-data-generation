import torchvision
import torch
import torch.optim as O

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.torch.Critic import Critic
from src.torch.Generator import Generator

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os
from src.torch.config import *
from src.torch.utils import initialize

ROOT_PATH = os.path.join("mnist_data")

criterion = torch.nn.BCELoss(reduction="sum")

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
    data = data.view(data.shape[0],-1)
    labels = mnist.targets

    # Taken by Rajarshee Mitra from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/10
    label_embedding = torch.nn.Embedding(NUMBER_OF_LABELS,NUMBER_OF_LABELS).to(gpu_device)
    label_embedding.weight.data = torch.eye(NUMBER_OF_LABELS).to(gpu_device)

    x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.1)

    x_train = x_train.to(gpu_device)
    y_train = y_train.to(gpu_device)
    x_test = x_test.to(gpu_device)
    y_test = y_test.to(gpu_device)

    generator = Generator(28,latent_size=LATENT_SIZE).to(gpu_device).apply(initialize)

    get_latent_variables = lambda batch_size=BATCH_SIZE: torch.randn((batch_size,LATENT_SIZE)).to(gpu_device)

    get_real_labels = lambda batch_size = BATCH_SIZE: 1 - torch.abs(torch.randn((batch_size,1))*0.05).to(gpu_device)
    get_fake_labels = lambda batch_size = BATCH_SIZE: torch.abs(torch.randn((batch_size,1))*0.05).to(gpu_device)


    critic = Critic(28,NUMBER_OF_LABELS).to(gpu_device).apply(initialize)

    critic_optimizer = O.Adam(critic.parameters(),lr=10e-5)
    generator_optimizer = O.Adam(generator.parameters(),lr=10e-5)

    train_results = torch.zeros((EPOCH, 7))
    test_train_results = torch.ones((TEST_BATCH, 1)).to(gpu_device)
    # Get [0 0 0 0 0 0 0 0 0 0 1 ..... 9 9 ] array
    test_image_labels = torch.arange(0, NUMBER_OF_LABELS).repeat(NUMBER_OF_LABELS).view(NUMBER_OF_LABELS, NUMBER_OF_LABELS).transpose(0, 1).reshape(-1,1).to(gpu_device)


    for epoch in trange(EPOCH):


        for step in range(STEPS_PER_EPOCH):

            for i in range(3):


                train_indices = torch.randint(0, len(x_train), (BATCH_SIZE,))
                train_batch = x_train[train_indices]
                label_batch = y_train[train_indices].view(BATCH_SIZE,1)

                real_output,label_output = critic(train_batch)
                classification_loss = criterion(label_output,label_embedding(label_batch).view(-1,NUMBER_OF_LABELS).detach())
                real_loss = criterion(real_output,get_real_labels())

                train_results[epoch, 0] += real_loss.item()
                train_results[epoch,1] += classification_loss.item()

                z = get_latent_variables()
                labels = torch.randint(0,9,(BATCH_SIZE,1)).to(gpu_device)
                fake_images = generator(z,labels)

                fake_output,label_output = critic(fake_images.detach())
                fake_loss = criterion(fake_output, get_fake_labels())

                critic.zero_grad()
                (classification_loss + real_loss + fake_loss).backward()
                critic_optimizer.step()

                train_results[epoch, 2] += fake_loss.item()


            z = get_latent_variables()
            labels = torch.randint(0, 9, (BATCH_SIZE, 1)).to(gpu_device)

            fake_images = generator(z,labels)
            fake_output,label_output = critic(fake_images)

            generator_classification_loss =  criterion(label_output,label_embedding(labels).view(-1,NUMBER_OF_LABELS).detach())
            generator_loss = criterion(fake_output, get_real_labels())

            generator.zero_grad()
            (generator_loss + generator_classification_loss).backward()
            generator_optimizer.step()

            train_results[epoch, 3] += generator_loss.item()
            train_results[epoch, 4] += generator_classification_loss.item()


            latent_variables = get_latent_variables(100)

            results = generator(latent_variables,test_image_labels)

            results_image = torchvision.utils.make_grid(results)

            torchvision.utils.save_image(results_image, os.path.join("results","results_step_{}.png".format(epoch*STEPS_PER_EPOCH)))

        test_indices = torch.randint(0, len(x_test), (TEST_BATCH,1))
        test_batch = x_test[test_indices]
        test_labels = y_test[test_indices].view(TEST_BATCH,1)

        test_output, test_class = critic(test_batch)
        test_loss = criterion(test_output,test_train_results)
        test_classification_loss = criterion(test_class,label_embedding(test_labels).view(-1,NUMBER_OF_LABELS).detach())

        train_results[epoch, 5] = test_classification_loss.item()
        train_results[epoch, 6] = test_loss.item()

        writer.add_scalar("real_train_loss",train_results[epoch,0]/STEPS_PER_EPOCH/5,epoch*STEPS_PER_EPOCH*5)
        writer.add_scalar("real_classification_loss",train_results[epoch,1]/STEPS_PER_EPOCH/5,epoch*STEPS_PER_EPOCH*5)
        writer.add_scalar("fake_train_loss",train_results[epoch,2]/STEPS_PER_EPOCH/5,epoch*STEPS_PER_EPOCH*5)
        writer.add_scalar("generator_loss",train_results[epoch,3]/STEPS_PER_EPOCH,epoch*STEPS_PER_EPOCH)
        writer.add_scalar("generator_classification_loss",train_results[epoch,4]/STEPS_PER_EPOCH,epoch*STEPS_PER_EPOCH)
        writer.add_scalar("test_classification_loss",train_results[epoch,5],epoch*STEPS_PER_EPOCH*5)
        writer.add_scalar("test_loss",train_results[epoch,6],epoch*STEPS_PER_EPOCH*5)


    latent_variables = get_latent_variables(TEST_IMAGES)
    results = generator(latent_variables)

    results_image = torchvision.utils.make_grid(results)

    torchvision.utils.save_image(results_image,"results.png")


    with open("generator.pkl", "wb") as fp:
        torch.save(generator, fp)

    with open("critic.pkl", "wb") as fp:
        torch.save(critic, fp)


    writer.close()