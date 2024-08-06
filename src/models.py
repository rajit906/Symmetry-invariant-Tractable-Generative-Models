import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc
from torch import distributions, nn
from util import RoundStraightThrough, log_discretized_logistic, log_mixture_discretized_logistic
from math import log, pi, exp
from scipy import linalg as la

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
            shape = (n_samples, 1, 28, 28) #self._c, self._h, self._w)
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


#####
# GLOW
#####
logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input