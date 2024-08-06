from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

def run(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glow trainer")
    parser.add_argument("--batch", default=16, type=int, help="batch size")
    parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
    parser.add_argument(
        "--n_flow", default=32, type=int, help="number of flows in each block"
    )
    parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument(
        "--affine", action="store_true", help="use affine coupling instead of additive"
    )
    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--img_size", default=64, type=int, help="image size")
    parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
    parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
    parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

    args = parser.parse_args()

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)