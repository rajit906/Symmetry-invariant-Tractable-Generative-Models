from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
"""Extra generative modeling benchmark datasets not provided by PyTorch."""

import os
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import PIL
import torch
from torch import distributions
from torch.nn import functional as F
from torch.utils import data
from util import translate_img_batch, translation_configurations

#Try binarize_torch from https://github.com/ajayjain/lmconv/blob/master/utils.py?

def _dynamically_binarize(x):
    return distributions.Bernoulli(probs=x).sample()

def _flatten(x):
    return (x.view(-1)).float()

def _flatten_and_multiply(x):
    return (255 * x.view(-1)).long()

def _gray_to_rgb(x):
        return x.repeat(3, 1, 1)

translation_repository = translation_configurations()

def _translate(x):
    sampled_translation = random.sample(translation_repository, 1)
    shift_left, shift_down, shift_right, shift_up = sampled_translation[0]
    return translate_img_batch(x.unsqueeze(dim=0), shift_left, shift_down, shift_right, shift_up)
    
def load_data(name, data_dir, binarize = False, eval = False, val = True, augment = False):
    """
    Loads the specified dataset and returns the training, validation, and test datasets.

    Args:
        name (str): The name of the dataset to load. Options are 'sklearn' and 'mnist'.

    Returns:
        tuple: A tuple containing the training dataset, validation dataset, and test dataset.
    """
    transform = [transforms.ToTensor(), transforms.Resize((28, 28)), _flatten]

    if binarize:
        transform.append(_dynamically_binarize)
    else:
        transform.append(_flatten_and_multiply)
    if eval:
        transform.remove(_flatten)
        if not binarize:
            transform.remove(_flatten_and_multiply)
        transform.append(transforms.Resize(299))
        transform.append(transforms.CenterCrop(299))
        transform.append(transforms.Grayscale(num_output_channels=3))
        transform.append(_gray_to_rgb) #transforms.Grayscale(num_output_channels=3))
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if augment:
        transform.append(_translate)
    transform = transforms.Compose(transform)
    if name == 'mnist':
        train_data = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
        test_data = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    if name == 'fashion':
        train_data = datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform)
        test_data = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform)
    if name == 'emnist':
        #train_data = datasets.EMNIST(root=data_dir, split='letters', train=True, download=False, transform=transform)
        train_data = None
        test_data = datasets.EMNIST(root=data_dir, split='letters', train=False, download=False, transform=transform)
    if name == 'omniglot':
        train_data = None
        #train_data = datasets.Omniglot(root=data_dir, train=True, download=True, transform=transform)
        test_data = datasets.Omniglot(root=data_dir, background=False, download=False, transform=transform)

    val_data = None
    if val:
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
        
    return (train_data, val_data, test_data)