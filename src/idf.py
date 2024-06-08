import torch
import torch.nn as nn
import torch.nn.functional as F
from util import RoundStraightThrough, log_integer_probability
import numpy as np


class IDF4(nn.Module):
    def __init__(self, netts, num_flows, D=2):
        super(IDF4, self).__init__()

        print('IDF by JT.')
        
        self.t_a = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
        self.t_b = torch.nn.ModuleList([netts[1]() for _ in range(num_flows)])
        self.t_c = torch.nn.ModuleList([netts[2]() for _ in range(num_flows)])
        self.t_d = torch.nn.ModuleList([netts[3]() for _ in range(num_flows)])
        
        self.num_flows = num_flows

        self.round = RoundStraightThrough.apply
        
        self.mean = nn.Parameter(torch.zeros(1, D))
        self.logscale = nn.Parameter(torch.ones(1, D))

        self.D = D

    def coupling(self, x, index, forward=True):

        (xa, xb, xc, xd) = torch.chunk(x, 4, 1)
        
        if forward:
            ya = xa + self.round(self.t_a[index](torch.cat((xb, xc, xd), 1)))
            yb = xb + self.round(self.t_b[index](torch.cat((ya, xc, xd), 1)))
            yc = xc + self.round(self.t_c[index](torch.cat((ya, yb, xd), 1)))
            yd = xd + self.round(self.t_d[index](torch.cat((ya, yb, yc), 1)))
        else:
            yd = xd - self.round(self.t_d[index](torch.cat((xa, xb, xc), 1)))
            yc = xc - self.round(self.t_c[index](torch.cat((xa, xb, yd), 1)))
            yb = xb - self.round(self.t_b[index](torch.cat((xa, yc, yd), 1)))
            ya = xa - self.round(self.t_a[index](torch.cat((yb, yc, yd), 1)))
        
        return torch.cat((ya, yb, yc, yd), 1)

    def permute(self, x):
        return x.flip(1)
    
    def log_prior(self, x):
        log_p = log_integer_probability(x, self.mean, self.logscale)
        return log_p.sum(1)

    def f(self, x):
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def forward(self, x, reduction='avg'):
        z = self.f(x)
        if reduction == 'sum':
            return -self.log_prior(z).sum()
        else:
            return -self.log_prior(z).mean()
        
    def prior_sample(self, batchSize, D=2):
        # Sample from logistic
        y = torch.rand(batchSize, self.D)
        x = torch.exp(self.logscale) * torch.log(y / (1. - y)) + self.mean
        # And then round it to an integer.
        return torch.round(x)

    def sample(self, batchSize):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=self.D)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batchSize, 1, self.D)