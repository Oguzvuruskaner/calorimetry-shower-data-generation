import torch
import torch.optim as O
import torch.nn as N

from scipy.stats import invgamma
from scipy.stats import uniform
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.config import *
from src.helpers.SphereDistribution import SphereDistribution
from src.models.EvaluationNetwork import EvaluationNetwork
from src.models.VariationNetwork import VariationNetwork

import os

from src.utils import create_or_cleanup


def sample_medium():
    def get_e():
        rv = invgamma(5, .05)
        return rv.rvs(size=1)

    get_sphere = SphereDistribution(3)(1)

    def wrapper() -> np.ndarray:
        e_values = []
        tmp_sum = 0

        while tmp_sum < 10:
            e = get_e()
            e_values.append(e)
            tmp_sum += e

        xyz_array = get_sphere(len(e_values))
        e_array = np.array(e_values).reshape(-1,1)

        return np.hstack([xyz_array,e_array])

    return wrapper

def sample_easy():

    get_sphere = SphereDistribution(3)(1)

    def get_e():
        rv = uniform(0, 1)
        return rv.rvs(size=1)

    def wrapper() -> np.ndarray:
        e_values = []
        tmp_sum = 0
        e = get_e()

        while tmp_sum < 10:
            e_values.append(e)
            e = get_e()
            tmp_sum += e

        e_values.append(10-sum(e_values))

        xyz_array = get_sphere(len(e_values))
        e_array = np.array(e_values).reshape(-1, 1)

        return np.hstack([xyz_array, e_array])

    return wrapper


def variation_init(m:N.Module):
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.uniform_(m.weight.data,-1e-8,1e-8)
        N.init.uniform_(m.bias.data, -1e-10,1e-10)


def evaluation_init(m:N.Module):

    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        N.init.kaiming_uniform_(m.weight.data)
        N.init.constant_(m.bias.data, 0)


def relaxed_main():


    GPU_DEVICE = torch.device(torch.cuda.current_device())
    MODEL_ROOT = os.path.join("..", "results", "particle_generator_training_sample_easy_{}".format(MODEL_VERSION))
    LOG_DIR = os.path.join("rnn_logs","train_sample_easy_{}".format(MODEL_VERSION))

    create_or_cleanup(MODEL_ROOT)
    create_or_cleanup(LOG_DIR)

    get_sample = sample_easy()

    net_var = VariationNetwork(4,1).apply(variation_init).to(GPU_DEVICE)
    net_eval = EvaluationNetwork(1).apply(evaluation_init).to(GPU_DEVICE)

    var_opt = O.Adam(net_var.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    var_scheduler = O.lr_scheduler.ExponentialLR(var_opt, gamma=0.98)

    eval_opt = O.Adam(net_var.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    eval_scheduler = O.lr_scheduler.ExponentialLR(eval_opt, gamma=0.95)

    writer = SummaryWriter(LOG_DIR)

    for epoch in trange(EPOCH):

        complete_train_results = 0
        complete_variational_results = 0

        eval_opt.zero_grad()
        var_opt.zero_grad()

        for step in range(STEPS_PER_EPOCH):
            for batch in range(BATCH_SIZE):

                jet = get_sample()
                jet = torch.Tensor(jet).to(GPU_DEVICE)
                length_of_jet = len(jet)
                label = torch.Tensor([0]).long().to(GPU_DEVICE)

                current_state = torch.zeros((1,STATE_SIZE),requires_grad=True).to(GPU_DEVICE)


                for ind,particles in enumerate(jet):

                    state = net_var(particles.view(1,4),label)
                    current_state += state.detach().sum(dim=0)

                    fake_result = net_eval(state,label)
                    fake_result.backward()


                    result = net_eval(current_state,label)
                    result.backward()

                for param in net_var.parameters():
                    if param.requires_grad:
                        param.grad /= 2*length_of_jet


                print(current_state.sum())
                eval_opt.step()
                result = -(net_eval(current_state,label))
                complete_train_results += int(result)
                result.backward()
                input_grad = [i.grad for i in net_eval.parameters()][0]
                input_grad /= input_grad.norm()

                for particles in jet:

                    state = net_var(particles.view(1,4),label)
                    state.backward(input_grad.view(1,-1))
                    tmp = state.detach()*input_grad.detach().sum(dim=0)
                    complete_variational_results += float(tmp.abs().sum())


                for param in net_var.parameters():
                    if param.requires_grad:
                        param.grad /= length_of_jet

                var_opt.step()
                eval_opt.step()

        var_scheduler.step()
        eval_scheduler.step()

        writer.add_scalar("Complete Train Loss",complete_train_results/BATCH_SIZE/STEPS_PER_EPOCH,BATCH_SIZE*STEPS_PER_EPOCH*(epoch+1))
        writer.add_scalar("Complete Variational Gradient L1",complete_variational_results/BATCH_SIZE/STEPS_PER_EPOCH,BATCH_SIZE*STEPS_PER_EPOCH*(epoch+1))

        torch.save(
            net_eval,
            os.path.join(MODEL_ROOT,"evaluator.tch")
        )

        torch.save(
            net_var,
            os.path.join(MODEL_ROOT, "variational.tch")
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

