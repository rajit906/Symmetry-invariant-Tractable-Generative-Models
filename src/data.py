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

translation_repository = translation_configurations()

def _translate(x):
    sampled_translation = random.sample(translation_repository, 1)
    shift_left, shift_down, shift_right, shift_up = sampled_translation[0]
    return translate_img_batch(x.unsqueeze(dim=0), shift_left, shift_down, shift_right, shift_up)
    
# Custom dataset class to return only images
class MNISTWithoutLabels(datasets.MNIST):
    """
    Custom dataset class for MNIST that returns only images, ignoring the labels.

    This class inherits from torchvision.datasets.MNIST and overrides the __getitem__ method to return only the images.
    """
    def __getitem__(self, index):
        """
        Returns the image at the specified index, ignoring the label.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            PIL.Image.Image: The image at the specified index.
        """
        img, _ = super().__getitem__(index)  # Ignore the label
        return img

def load_data(name, data_dir, binarize = False, eval = False, val = True, augment = False):
    """
    Loads the specified dataset and returns the training, validation, and test datasets.

    Args:
        name (str): The name of the dataset to load. Options are 'sklearn' and 'mnist'.

    Returns:
        tuple: A tuple containing the training dataset, validation dataset, and test dataset.
    """
    transform = [transforms.ToTensor()]

    #if name == 'sklearn':
        #train_data = Digits(mode='train')
        #val_data = Digits(mode='val')
        #test_data = Digits(mode='test')

    if name == 'mnist':
        if binarize:
            transform.append(_dynamically_binarize)
            transform.append(_flatten)
        else:
            transform.append(_flatten_and_multiply)
        if eval:
            transform.remove(_flatten)
            transform.append(transforms.Resize((299, 299)))
            transform.append(transforms.Grayscale(num_output_channels=3))
            transform.append(transforms.Normalize((0.5,), (0.5,)))
        if augment:
            transform.append(_translate)

        transform = transforms.Compose(transform)
        train_data = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
        test_data = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
        #train_data = MNISTWithoutLabels(root=data_dir, train=True, download=True, transform=transform)
        #test_data = MNISTWithoutLabels(root=data_dir, train=False, download=True, transform=transform)

    val_data = None
    if val:
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])

        
    return (train_data, val_data, test_data)