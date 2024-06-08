import torch
import torch.nn as nn

# The general invertible transformation in (Tomczak, 2020) with 4 partitions
def nnetts(D, M):
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