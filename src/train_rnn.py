from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.config import *
from src.datasets import DATASETS
from src.helpers.JetDataset import JetDataset
from src.models.EvaluationNetwork import EvaluationNetwork
from src.models.VariationNetwork import VariationNetwork
import os

import torch.nn as N
import torch.optim as O
import torch

from src.utils import bernoulli, create_or_cleanup
from math import log2


def variation_init(m:N.Module):
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data)
        N.init.constant_(m.bias.data, 0)


def evaluation_init(m:N.Module):

    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data)
        N.init.constant_(m.bias.data, 0)

GPU_DEVICE = torch.device(torch.cuda.current_device())


def entropy_loss(information_prob):

    def wrapper(output,correct):
        return -log2(information_prob) * (output-correct)**2

    return wrapper

def group_list(l,group_size):
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]


if __name__ == "__main__":

    train_dataset = JetDataset(os.path.join("..","data","particle_dataset","all.h5"))


    MODEL_ROOT = os.path.join("..", "results", "particle_generator_training_{}".format(MODEL_VERSION))
    LOG_DIR = os.path.join("rnn_logs","train_{}".format(MODEL_VERSION))


    create_or_cleanup(MODEL_ROOT)
    create_or_cleanup(LOG_DIR)
    writer = SummaryWriter(LOG_DIR)

    get_jet = lambda : next(iter(DataLoader(train_dataset,shuffle=True,batch_size=1)))

    number_of_labels = len(DATASETS.keys())

    gev_to_embed =  {}

    net_var = VariationNetwork(4,number_of_labels).apply(variation_init).to(GPU_DEVICE)
    net_eval = EvaluationNetwork(number_of_labels).apply(evaluation_init).to(GPU_DEVICE)

    var_opt = O.Adam(net_var.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    var_scheduler = O.lr_scheduler.ExponentialLR(var_opt,gamma=0.98)

    eval_opt = O.Adam(net_var.parameters(), lr=LEARNING_RATE*40, weight_decay=WEIGHT_DECAY)
    eval_scheduler = O.lr_scheduler.ExponentialLR(eval_opt, gamma=0.95)

    for ind,key in enumerate(DATASETS.keys()):
        gev_to_embed[key] = ind


    for epoch in trange(EPOCH):

        complete_train_results = 0
        complete_variational_results = 0

        for step in range(STEPS_PER_EPOCH):
            for batch in range(BATCH_SIZE):

                jet,GeV = get_jet()
                jet = jet.view(-1,4).to(GPU_DEVICE)
                length_of_jet = len(jet)
                label = gev_to_embed[int(GeV)]


                fake_criterion = entropy_loss((length_of_jet-1)/length_of_jet)
                real_criterion = entropy_loss(1/length_of_jet)

                current_state = torch.zeros((1,STATE_SIZE),requires_grad=True).to(GPU_DEVICE)
                labels_arr_1 = torch.Tensor(GROUP_SIZE*[label]).long().to(GPU_DEVICE)
                labels_arr_2 = torch.Tensor((length_of_jet%GROUP_SIZE)*[label]).long().to(GPU_DEVICE)


                for particles in group_list(jet,GROUP_SIZE):

                    if len(particles) == GROUP_SIZE:
                        labels = labels_arr_1
                    else:
                        labels = labels_arr_2

                    state = net_var(particles.view(-1,4),labels)
                    current_state += state.detach().sum(dim=0)

                    result = net_eval(current_state,labels[0])
                    result = fake_criterion(result,0)
                    result.backward()

                result = real_criterion(net_eval(current_state,labels[0]),1) + STATE_DECAY * current_state.sum()
                complete_train_results += int(result)
                result.backward()

                input_grad = [i.grad for i in net_eval.parameters()][0]

                for particles in group_list(jet,GROUP_SIZE):
                    if len(particles) == GROUP_SIZE:
                        labels = labels_arr_1
                    else:
                        labels = labels_arr_2

                    state = net_var(particles.view(-1,4),labels)
                    state = state.sum(dim=0).view(1,-1)
                    state.backward(input_grad.view(1,-1))
                    tmp = state.detach()*input_grad.detach().sum(dim=0)
                    complete_variational_results += float(tmp.abs().sum())



            for param in net_var.parameters():
                if param.requires_grad:
                    param.grad /= BATCH_SIZE

            eval_opt.step()
            var_opt.step()

        var_scheduler.step()
        eval_scheduler.step()

        writer.add_scalar("Complete Train Loss",complete_train_results/BATCH_SIZE/STEPS_PER_EPOCH,BATCH_SIZE*STEPS_PER_EPOCH*(epoch+1))
        writer.add_scalar("Complete Variational Gradient L1",complete_variational_results/BATCH_SIZE/STEPS_PER_EPOCH,BATCH_SIZE*STEPS_PER_EPOCH*(epoch+1))

        torch.save(
            net_eval,
            os.path.join("..","results","particle_generator_training_{}".format(MODEL_VERSION),"evaluator.tch")
        )

        torch.save(
            net_var,
            os.path.join("..", "results", "particle_generator_training_{}".format(MODEL_VERSION), "variational.tch")
        )

        if not epoch % CHECKPOINT_RATE:
            torch.save(
                net_eval,
                os.path.join("..", "results", "particle_generator_training_{}".format(MODEL_VERSION), "evaluator_{}.tch".format(epoch))
            )
            torch.save(
                net_var,
                os.path.join("..", "results", "particle_generator_training_{}".format(MODEL_VERSION),
                             "variational_{}.tch".format(epoch))
            )