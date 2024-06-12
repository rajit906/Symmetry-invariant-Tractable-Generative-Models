import torch
import torch.nn as nn
import torch.nn.functional as F
from util import RoundStraightThrough, log_discretized_logistic, log_mixture_discretized_logistic


class IDF4(nn.Module):
    """
    An implementation of the IDF (Integer Discrete Flow) model.

    This class defines a flow-based generative model designed to work with discrete integer data.
    The model uses coupling layers and a series of transformations to model the data distribution.
    
    Attributes:

        t_a (torch.nn.ModuleList): A list of transformation networks for the 'a' component.
        t_b (torch.nn.ModuleList): A list of transformation networks for the 'b' component.
        t_c (torch.nn.ModuleList): A list of transformation networks for the 'c' component.
        t_d (torch.nn.ModuleList): A list of transformation networks for the 'd' component.
        num_flows (int): The number of flow steps in the model.
        round (function): A function for rounding with straight-through gradient estimation.
        mean (torch.nn.Parameter): The mean parameter of the prior distribution.
        logscale (torch.nn.Parameter): The log-scale parameter of the prior distribution.
        D (int): The dimensionality of the input data.
    """
    def __init__(self, netts, num_flows, n_mixtures = 1, D = 2):
        """
        Initializes the IDF4 model.

        Args:
            netts (list): A list of functions to create transformation networks.
            num_flows (int): The number of flow steps.
            D (int): The dimensionality of the input data (default is 2).
        """
        super(IDF4, self).__init__()
        self.round = RoundStraightThrough.apply
        self.num_flows = num_flows
        self.D = D
        self.n_mixtures = n_mixtures
        if self.n_mixtures == 1:
            self.mean = nn.Parameter(torch.zeros(1, D))
            self.logscale = nn.Parameter(torch.ones(1, D))
            self.is_mixture = False
        elif self.n_mixtures > 1:
            self.means = nn.Parameter(torch.zeros(1, D, self.n_mixtures))
            self.logscales = nn.Parameter(torch.ones(1, D, self.n_mixtures))
            self.pi_logit = nn.Parameter(torch.ones(1, D, self.n_mixtures) / self.n_mixtures)
            self.is_mixture = True

        
        self.t_a = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
        self.t_b = torch.nn.ModuleList([netts[1]() for _ in range(num_flows)])
        self.t_c = torch.nn.ModuleList([netts[2]() for _ in range(num_flows)])
        self.t_d = torch.nn.ModuleList([netts[3]() for _ in range(num_flows)])

    def coupling(self, x, index, forward=True):
        """
        Applies the coupling layer transformations.

        Args:
            x (torch.Tensor): The input tensor to transform.
            index (int): The index of the flow step.
            forward (bool): Direction of transformation. True for forward, False for inverse (default is True).

        Returns:
            torch.Tensor: The transformed tensor.
        """

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
        """
        Permutes the input tensor by flipping it along the last dimension.

        Args:
            x (torch.Tensor): The input tensor to permute.

        Returns:
            torch.Tensor: The permuted tensor.
        """
        return x.flip(1)
    
    def log_prior(self, x):
        """
        Computes the log-prior probability of the given tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The log-prior probability of the input tensor.
        """
        if self.is_mixture:
            self.pi = F.softmax(self.pi_logit, dim=-1)
            x = x.unsqueeze(-1).repeat(1, 1, self.n_mixtures)
            log_p = log_mixture_discretized_logistic(x, self.means, self.logscales, self.pi)
        else:
            log_p = log_discretized_logistic(x, self.mean, self.logscale)
        return log_p.sum(1)

    def f(self, x):
        """
        Applies the forward flow transformations.

        Args:
            x (torch.Tensor): The input tensor to transform.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        """
        Applies the inverse flow transformations.

        Args:
            z (torch.Tensor): The transformed tensor to invert.

        Returns:
            torch.Tensor: The inverted tensor.
        """
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def forward(self, x, reduction='avg'):
        """
        Computes the forward pass of the model and the negative log-prior.

        Args:
            x (torch.Tensor): The input tensor.
            reduction (str): The reduction method to apply to the log-prior ('sum' or 'avg', default is 'avg').

        Returns:
            torch.Tensor: The negative log-prior of the transformed tensor.
        """
        z = self.f(x)
        if reduction == 'sum':
            return -self.log_prior(z).sum()
        else:
            return -self.log_prior(z).mean()
        
    def prior_sample(self, batchSize, D=2, inverse_bin_width = 1.):
        """
        Samples from the prior (Logistic, Discretized Logistic, Discretized Logistic Mixture) distribution.

        Args:
            batchSize (int): The number of samples to generate.
            D (int): The dimensionality of the data (default is 2).

        Returns:
            torch.Tensor: The sampled tensor from the prior distribution.
        """
        if self.is_mixture:
            _, _, n_mixtures = tuple(map(int, self.pi.size()))
            pi = self.pi.view(D, n_mixtures).clone()
            sampled_pi = torch.multinomial(pi, num_samples=1).view(-1)

            # Select mixture params
            means = self.means.view(D, n_mixtures)
            means = means[torch.arange(D), sampled_pi].view(D)
            logscales = self.logscales.view(D, n_mixtures).clone()
            logscales = logscales[torch.arange(D), sampled_pi].view(D)

            y = torch.rand_like(means)
            x = (torch.exp(logscales) * torch.log(y / (1 - y)) + means).view(1, D)

        else:
            y = torch.rand(batchSize, self.D)
            x = torch.exp(self.logscale) * torch.log(y / (1. - y)) + self.mean
            
        return torch.round(x * inverse_bin_width) / inverse_bin_width

    def sample(self, batchSize):
        """
        Generates samples from the model.

        Args:
            batchSize (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated samples.
        """
        z = self.prior_sample(batchSize=batchSize, D=self.D)
        x = self.f_inv(z) # = f^-1(z)
        return x.view(batchSize, 1, self.D)