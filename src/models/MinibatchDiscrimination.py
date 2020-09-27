import torch
import torch.nn as N


class MinibatchDiscrimination(torch.nn.Module):

    """
        Improved Techniques for Training GANs
        https://arxiv.org/pdf/1606.03498.pdf

        T : A x B x C tensor
        input_dim : It corresponds to A variable. It is the size of the features.
        out_dim : It corresponds to B variable.

    """
    def __init__(self,input_dim:int,out_dim:int,C:int):
        super().__init__()

        self.T = N.Parameter(torch.zeros([input_dim,out_dim,C]))
        N.init.kaiming_normal_(self.T)
        self.A = input_dim
        self.B = out_dim
        self.C = C

    def forward(self,x):

        M = x.mm(self.T.view(self.A,-1))

        #N x B x C
        M = M.view(-1,self.B,self.C)
        O = torch.zeros(M.shape[0],self.B,requires_grad=False).to(x.device).detach()

        for i in range(x.shape[0]):
            O[i,:] = torch.exp(-torch.abs(M-M[i]).sum(2)).sum(0)

        return O


