import torch.nn as N
import torch


class NFilter(N.Module):

    def __init__(self,N_in,N_out):
        super().__init__()

        self.N_in = N_in
        self.N_out = N_out

        self.model = N.Sequential(
            N.Conv1d(N_in, 16, 5, 1, 2),
            N.BatchNorm1d(16),
            N.LeakyReLU(),
            N.Conv1d(16, 16, 5, 1, 2),
            N.BatchNorm1d(16),
            N.LeakyReLU(),
            N.Conv1d(16, 16, 5, 1, 2),
            N.BatchNorm1d(16),
            N.LeakyReLU(),
            N.Conv1d(16, N_out, 5, 1, 2),
            N.Sigmoid()
        )

    def forward(self, *args):
        x = torch.cat(args, dim=1)
        return self.model(x)


