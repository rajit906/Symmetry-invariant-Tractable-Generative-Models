import torch
import torch.nn as nn

# The general invertible transformation in (Tomczak, 2020) with 4 partitions
def nnetts(D, M):
    """
    Creates a list of neural network generator functions for the IDF4 model (Tomcak, 2020).

    Each function generates a neural network with the specified architecture. The networks are 
    designed to process different partitions of the input data within the IDF4 model.

    Args:
        D (int): The dimensionality of the input data.
        M (int): The number of hidden units in the intermediate layers.

    Returns:
        list: A list of functions, each of which generates a neural network with a specific architecture.
    """
    nett_a = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, D // 4))

    nett_b = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                        nn.Linear(M, M), nn.LeakyReLU(),
                                        nn.Linear(M, D // 4))

    nett_c = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                        nn.Linear(M, M), nn.LeakyReLU(),
                                        nn.Linear(M, D // 4))

    nett_d = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                        nn.Linear(M, M), nn.LeakyReLU(),
                                        nn.Linear(M, D // 4))

    netts = [nett_a, nett_b, nett_c, nett_d]
    return netts