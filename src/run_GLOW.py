import numpy as np
import argparse

import torch
from torch import nn, optim
from models import Glow
from train_GLOW import train, evaluation

def run(args):
    n_flow = args.n_flow
    n_block = args.n_block
    affine = args.affine
    no_lu = args.no_lu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_single = Glow(
        3, n_flow, n_block, affine=affine, conv_lu=not no_lu
    )
    model = nn.DataParallel(model_single)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer)

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
    run(args)