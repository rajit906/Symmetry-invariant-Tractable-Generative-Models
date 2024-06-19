import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
from torch import distributions, nn
from util import RoundStraightThrough, log_discretized_logistic, log_mixture_discretized_logistic

"""Base classes for models."""



def _default_sample_fn(logits):
    return distributions.Bernoulli(logits=logits).sample()


def auto_reshape(fn):
    """Decorator which flattens image inputs and reshapes them before returning.

    This is used to enable non-convolutional models to transparently work on images.
    """

    def wrapped_fn(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        y = fn(self, x, *args, **kwargs)
        return y.view(original_shape)

    return wrapped_fn


class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.

    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    def __call__(self, x, *args, **kwargs):
        """Saves input tensor attributes so they can be accessed during sampling."""
        if getattr(self, "_c", None) is None and x.dim() == 4:
            _, c, h, w = x.shape
            self._create_shape_buffers(c, h, w)
        return super().__call__(x, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Registers dynamic buffers before loading the model state."""
        if "_c" in state_dict and not getattr(self, "_c", None):
            c, h, w = state_dict["_c"], state_dict["_h"], state_dict["_w"]
            self._create_shape_buffers(c, h, w)
        super().load_state_dict(state_dict, strict)

    def _create_shape_buffers(self, channels, height, width):
        channels = channels if torch.is_tensor(channels) else torch.tensor(channels)
        height = height if torch.is_tensor(height) else torch.tensor(height)
        width = width if torch.is_tensor(width) else torch.tensor(width)
        self.register_buffer("_c", channels)
        self.register_buffer("_h", height)
        self.register_buffer("_w", width)

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples):
        ...


class AutoregressiveModel(GenerativeModel):
    """The base class for Autoregressive generative models."""

    def __init__(self, sample_fn=None):
        """Initializes a new AutoregressiveModel instance.

        Args:
            sample_fn: A fn(logits)->sample which takes sufficient statistics of a
                distribution as input and returns a sample from that distribution.
                Defaults to the Bernoulli distribution.
        """
        super().__init__()
        self._sample_fn = sample_fn or _default_sample_fn

    def _get_conditioned_on(self, n_samples, conditioned_on):
        assert (
            n_samples is not None or conditioned_on is not None
        ), 'Must provided one, and only one, of "n_samples" or "conditioned_on"'
        if conditioned_on is None:
            shape = (n_samples, self._c, self._h, self._w)
            conditioned_on = (torch.ones(shape) * -1).to(self.device)
        else:
            conditioned_on = conditioned_on.clone()
        return conditioned_on

    @torch.no_grad()
    def sample(self, n_samples=None, conditioned_on=None):
        """Generates new samples from the model.

        Args:
            n_samples: The number of samples to generate. Should only be provided when
                `conditioned_on is None`.
            conditioned_on: A batch of partial samples to condition the generation on.
                Only dimensions with values < 0 are sampled while dimensions with
                values >= 0 are left unchanged. If 'None', an unconditional sample is
                generated.
        """
        conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
        n, c, h, w = conditioned_on.shape
        for row in range(h):
            for col in range(w):
                out = self.forward(conditioned_on)[:, :, row, col]
                out = self._sample_fn(out).view(n, c)
                conditioned_on[:, :, row, col] = torch.where(
                    conditioned_on[:, :, row, col] < 0,
                    out,
                    conditioned_on[:, :, row, col],
                )
        return conditioned_on

class MaskedLinear(nn.Linear):
    """A Linear layer with masks that turn off some of the layer's weights."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones((out_features, in_features)))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class MADE(AutoregressiveModel):
    """The Masked Autoencoder Distribution Estimator (MADE) model."""

    def __init__(self, input_dim, hidden_dims=None, n_masks=1, sample_fn=None):
        """Initializes a new MADE instance.

        Args:
            input_dim: The dimensionality of the input.
            hidden_dims: A list containing the number of units for each hidden layer.
            n_masks: The total number of distinct masks to use during training/eval.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input_dim = input_dim
        self._dims = [self._input_dim] + (hidden_dims or []) + [self._input_dim]
        self._n_masks = n_masks
        self._mask_seed = 0

        layers = []
        for i in range(len(self._dims) - 1):
            in_dim, out_dim = self._dims[i], self._dims[i + 1]
            layers.append(MaskedLinear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self._net = nn.Sequential(*layers[:-1])

    def _sample_masks(self):
        """Samples a new set of autoregressive masks.

        Only 'self._n_masks' distinct sets of masks are sampled after which the mask
        sets are rotated through in the order in which they were sampled. In
        principle, it's possible to generate the masks once and cache them. However,
        this can lead to memory issues for large 'self._n_masks' or models many
        parameters. Finally, sampling the masks is not that computationally
        expensive.

        Returns:
            A tuple of (masks, ordering). Ordering refers to the ordering of the outputs
            since MADE is order agnostic.
        """
        rng = np.random.RandomState(seed=self._mask_seed % self._n_masks)
        self._mask_seed += 1

        # Sample connectivity patterns.
        conn = [rng.permutation(self._input_dim)]
        for i, dim in enumerate(self._dims[1:-1]):
            # NOTE(eugenhotaj): The dimensions in the paper are 1-indexed whereas
            # arrays in Python are 0-indexed. Implementation adjusted accordingly.
            low = 0 if i == 0 else np.min(conn[i - 1])
            high = self._input_dim - 1
            conn.append(rng.randint(low, high, size=dim))
        conn.append(np.copy(conn[0]))

        # Create masks.
        masks = [
            conn[i - 1][None, :] <= conn[i][:, None] for i in range(1, len(conn) - 1)
        ]
        masks.append(conn[-2][None, :] < conn[-1][:, None])

        return [torch.from_numpy(mask.astype(np.uint8)) for mask in masks], conn[-1]

    def _forward(self, x, masks):
        layers = [
            layer for layer in self._net.modules() if isinstance(layer, MaskedLinear)
        ]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)
        return self._net(x)

    @auto_reshape
    def forward(self, x):
        """Computes the forward pass.

        Args:
            x: Either a tensor of vectors with shape (n, input_dim) or images with shape
                (n, 1, h, w) where h * w = input_dim.
        Returns:
            The result of the forward pass.
        """

        masks, _ = self._sample_masks()
        return self._forward(x, masks)

    @torch.no_grad()
    def sample(self, n_samples, conditioned_on=None):
        """See the base class."""
        conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
        return self._sample(conditioned_on)

    @auto_reshape
    def _sample(self, x):
        masks, ordering = self._sample_masks()
        ordering = np.argsort(ordering)
        for dim in ordering:
            out = self._forward(x, masks)[:, dim]
            out = self._sample_fn(out)
            x[:, dim] = torch.where(x[:, dim] < 0, out, x[:, dim])
        return x



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