import os
import json
import random
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import models
import wandb
from train import evaluation, training 
from data import load_data
import numpy as np
from util import categorical_layer_factory, hadamard_layer_factory, dense_layer_factory, mixing_layer_factory, cross_entropy_loss_fn

from Cirkits.cirkit.templates.region_graph import QuadTree
from Cirkits.cirkit.symbolic.circuit import Circuit
from Cirkits.cirkit.pipeline import PipelineContext



def run(args):
    random.seed(42)
    np.random.seed(42) #Remove this if you are doing several runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    result_dir = args.model_path
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)
    lam = args.lam
    patience = args.patience
    input_dim = args.input_dim
    data = args.data
    binarize_flag = args.binarize
    train_data, val_data, test_data = load_data(data, binarize = binarize_flag)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    if model_type == 'MADE':
        n_masks = args.num_masks
        M = args.hidden_layers
        ### Naming Folder
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        hyperparams = f"_E{num_epochs}_BS{batch_size}_LR{lr:.0e}_M{M}_NM{n_masks}_ID{input_dim}_{data}"
        if args.binarize:
            hyperparams += "_BIN"
        hyperparams += f"_LAM{lam}_PAT{patience}"
        name = f"{current_time}_{hyperparams}"
        if not(os.path.exists(result_dir + '/' + name)):
            os.mkdir(result_dir + '/' + name)
        ### Define Model
        model = models.MADE(input_dim=input_dim, hidden_dims=[M], n_masks=n_masks).to(device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr = lr)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)
        nll_val, bpd_val, nll_train, model = training(name=name, result_dir=result_dir, model_type = model_type, max_patience=patience, num_epochs=num_epochs, 
                           model=model, loss_fn=cross_entropy_loss_fn, optimizer=optimizer, lam=lam,
                           training_loader=train_loader, val_loader=val_loader, device=device, batch_size = batch_size)
        torch.save(model, result_dir + '/' + name + '/model_best.model')

    elif model_type == 'PC':
        num_input_units = args.num_input_units
        num_sum_units = args.num_sum_units
        height = args.height

        ### Naming Folder
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        hyperparams = f"_E{num_epochs}_BS{batch_size}_LR{lr:.0e}_H{height}_NSM{num_sum_units}_NIM{num_input_units}_ID{input_dim}_{data}"
        if args.binarize:
            hyperparams += "_BIN"
        hyperparams += f"_LAM{lam}_PAT{patience}"
        name = f"{current_time}_{hyperparams}"
        if not(os.path.exists(result_dir + '/' + name)):
            os.mkdir(result_dir + '/' + name)
        ### Define Model
        region_graph = QuadTree(shape=(height, height))

        symbolic_circuit = Circuit.from_region_graph(
            region_graph,
            num_input_units=num_input_units,
            num_sum_units=num_sum_units,
            input_factory=categorical_layer_factory,
            sum_factory=dense_layer_factory,
            prod_factory=hadamard_layer_factory,
            mixing_factory=mixing_layer_factory
        )

        ctx = PipelineContext(
            backend='torch',   # Choose the torch compilation backend
            fold=True,         # Fold the circuit, this is a backend-specific compilation flag
            semiring='lse-sum' # Use the (R, +, *) semiring, where + is the log-sum-exp and * is the sum
        )
        circuit = ctx.compile(symbolic_circuit).to(device)
        pf_circuit = ctx.integrate(circuit).to(device)
        model = (circuit, pf_circuit)

        optimizer = torch.optim.SGD([p for p in circuit.parameters() if p.requires_grad == True], lr=lr, momentum=0.95) #torch.optim.Adam([p for p in circuit.parameters() if p.requires_grad == True], lr = lr)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)

        nll_val, bpd_val, nll_train, model = training(name=name, result_dir=result_dir, model_type = model_type, max_patience=patience, 
                                                      num_epochs=num_epochs, model=model, loss_fn=cross_entropy_loss_fn, optimizer=optimizer, lam=lam,
                                                      training_loader=train_loader, val_loader=val_loader, device=device, batch_size = batch_size)
        circuit, pf_circuit = model
        torch.save(circuit, result_dir + '/' + name + '/circuit.pt')
        torch.save(pf_circuit, result_dir + '/' + name + '/pfcircuit.pt')

    test_val, test_bpd = evaluation(test_loader, device, model_type = model_type, model_best=model)
    _, _, aug_test_data = load_data(data, binarize = binarize_flag, augment = True, val = False)
    aug_test_loader = DataLoader(aug_test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    aug_test_val, aug_test_bpd = evaluation(aug_test_loader, device, model_type = model_type, model_best=model)

    results_dic = {'nll_val': nll_val.tolist(), 
                    'bpd_val': bpd_val.tolist(), 
                    'nll_train': nll_train.tolist(),
                    'test_val': test_val,
                    'test_bpd': test_bpd,
                    'aug_test_bpd': aug_test_bpd,
                    'aug_test_val': aug_test_val,
                    'KID': 'N/A'}
        
    with open(result_dir + '/' + name + '/results.json', 'w') as json_file:
        json.dump(results_dic, json_file, indent=4)
    print(f"test_bpd = {test_bpd}", f"test_nll = {test_val}", f"aug_test_bpd = {aug_test_bpd}", f"aug_test_nll = {aug_test_val}")

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Generative Model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['PC', 'MADE'], help='Type of model to train')
    args, remaining_args = parser.parse_known_args()

    if args.model_type == 'PC':
        parser.add_argument('--num_input_units', type=int, default=8, help='Number of Input Units (sum, input)')
        parser.add_argument('--num_sum_units', type=int, default=8, help='Number of Input Units (sum, input)')
        parser.add_argument('--height', type=int, default=8, help='Height')

    elif args.model_type == 'MADE':
        parser.add_argument('--hidden_layers', type=int, default=8000, help='Number of layers for the hidden layer')
        parser.add_argument('--num_masks', type=int, default=1, help='Number of Masks')

    parser.add_argument('--input_dim', type=int, default=784, help='Input Dimension')
    parser.add_argument('--data', type=str, choices=['mnist'], help='Dataset to use')
    parser.add_argument('--binarize', action='store_true', help='Flag to use binarized data or not')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the trained model')
    parser.add_argument('--lam', type=float, default=1.0, help='Regularization Hyperparameter')
    parser.add_argument('--patience', type=int, default=20, help='Early Stopping Mechanism')

    args = parser.parse_args()
    run(args)
