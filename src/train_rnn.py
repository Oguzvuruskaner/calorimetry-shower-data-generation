from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.config import *
from src.helpers.JetDataset import JetDataset
import os

import torch.nn as N
import torch.optim as O
import torch

from src.models.RNN import RNN
from src.plots import plot_multiple_images, plot_energy_graph, get_jet_images
from src.utils import create_or_cleanup

import matplotlib.pyplot as plt


GPU_DEVICE = torch.device(torch.cuda.current_device())

def gate_init(m:N.Module):
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data)
        N.init.zeros_(m.bias.data)


def network_init(m:N.Module):
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_normal_(m.weight.data)
        N.init.zeros_(m.bias.data)


MODEL_VERSION = 1
TRAIN_LABEL = 1
HARD_STOP = 1000


if __name__ == "__main__":

    train_dataset = JetDataset(os.path.join("..","data","particle_dataset",str(TRAIN_LABEL),"all.h5"))

    RESULTS_DIR = os.path.join("..", "results", "sequential_training_{}".format(MODEL_VERSION))
    LOG_DIR = os.path.join("../logs/rnn_logs", "sequential_training_{}".format(MODEL_VERSION))
    MODELS_ROOT_DIR = os.path.join("..","models","sequential_training_{}".format(MODEL_VERSION))
    
    

    create_or_cleanup(RESULTS_DIR)
    create_or_cleanup(LOG_DIR)
    create_or_cleanup(MODELS_ROOT_DIR)
    writer = SummaryWriter(LOG_DIR)

    plot_func = lambda data, title, ax: plot_energy_graph(data, title, TRAIN_LABEL, ax=ax)

    get_jet = lambda : next(iter(DataLoader(train_dataset,shuffle=True,batch_size=1)))

    criterion = torch.nn.MSELoss()

    model = RNN().to(GPU_DEVICE)

    optim = O.ASGD(model.parameters(),lr=LEARNING_RATE)
    lr_scheduler = O.lr_scheduler.ExponentialLR(optim,0.96)



    for epoch in trange(EPOCH):

        training_loss = 0

        for steps in range(STEPS_PER_EPOCH):

            particles,_ = get_jet()
            particles = particles.view(-1,4)

            c = torch.zeros(STATE_SIZE).to(GPU_DEVICE)
            h = torch.rand(STATE_SIZE).to(GPU_DEVICE)

            for particle in particles:

                out_particle,c,h = model(c.detach(),h.detach())

                loss = criterion(out_particle,particle.to(GPU_DEVICE))
                training_loss += loss.item()

                loss.backward()


            end_particle,_,_ = model(c.detach(),h.detach())
            loss = criterion(end_particle,torch.zeros(4).to(GPU_DEVICE))
            training_loss += loss.item()

            loss.backward()
            optim.step()


        lr_scheduler.step()

        if not epoch % CHECKPOINT_RATE:
            torch.save(model,os.path.join(MODELS_ROOT_DIR,"lstm_{}.pt".format(epoch)))

        generated_jets = torch.zeros((TEST_IMAGES,HARD_STOP,3))


        for jet_no in range(TEST_IMAGES):

            c = torch.zeros(STATE_SIZE).to(GPU_DEVICE)
            h = torch.rand(STATE_SIZE).to(GPU_DEVICE)

            #No time for caution
            for ind in range(HARD_STOP):

                particle,c,h = model(c.detach(),h.detach())

                if particle.sum() != 0:
                    generated_jets[jet_no, ind] = torch.FloatTensor([
                            torch.sqrt(particle[0]*particle[0] + particle[1]*particle[1]),
                            particle[2],
                            particle[3]
                     ])

                else:
                    break

        images = get_jet_images(generated_jets.numpy())
        fig = plot_multiple_images(images,4)

        fig.savefig(os.path.join(RESULTS_DIR, "{}.png".format((epoch+1)*STEPS_PER_EPOCH)))
        plt.close(fig)

        plot_func = lambda data,title,ax : plot_energy_graph(data,title,TRAIN_LABEL,ax=ax)
        #Only the energy values are needed AND t o, therefore,
        fig = plot_multiple_images(generated_jets[:,:,-2:-1].numpy(),4,plot_func=plot_func)
        fig.savefig(os.path.join(RESULTS_DIR, "{}_energy.png".format((epoch+1)*STEPS_PER_EPOCH)))
        plt.close(fig)

        writer.add_scalar("Train Error", training_loss / STEPS_PER_EPOCH,
                          (epoch+1) * STEPS_PER_EPOCH)
