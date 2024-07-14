import os
import json
from datetime import datetime
import random
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import wandb
from train_PC import evaluation, training 
from data import load_data
import numpy as np
from util import categorical_layer_factory, hadamard_layer_factory, dense_layer_factory, mixing_layer_factory

from Cirkits.cirkit.templates.region_graph import QuadTree
from Cirkits.cirkit.symbolic.circuit import Circuit
from Cirkits.cirkit.pipeline import PipelineContext

def run(args):
    random.seed(42)
    np.random.seed(42) #Remove this if you are doing several runs
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    num_input_units = args.num_input_units
    num_sum_units = args.num_sum_units
    height = args.height
    config = {'input_dim': input_dim, 'lr': lr, 'num_epochs': num_epochs, 'max_patience': patience, 'batch_size': batch_size, 'lambda': lam, 'num_input_units': num_input_units, 'num_sum_units': num_sum_units}
    #run = wandb.init(entity="rajpal906")#entity="rajpal906", project="MADE", name="unregularized", id="1", config=hyperparameters, settings=wandb.Settings(start_method="fork"))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparams = f"_E{num_epochs}_BS{batch_size}_LR{lr:.0e}_H{height}_NSM{num_sum_units}_NIM{num_input_units}_ID{input_dim}_{data}"
    if args.binarize:
        hyperparams += "_BIN"
    hyperparams += f"_LAM{lam}_PAT{patience}"
    name = f"{current_time}_{hyperparams}"
    if not(os.path.exists(result_dir + '/' + name)):
        os.mkdir(result_dir + '/' + name)

    with open(result_dir + '/' + name + '/config.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)

    region_graph = QuadTree(shape=(height, height))
    symbolic_circuit = Circuit.from_region_graph(
    region_graph,
    num_input_units=num_input_units,
    num_sum_units=num_sum_units,
    input_factory=categorical_layer_factory,
    sum_factory=dense_layer_factory,
    prod_factory=hadamard_layer_factory,
    mixing_factory=mixing_layer_factory)

    ctx = PipelineContext(
        backend='torch',   # Choose the torch compilation backend
        fold=True,         # Fold the circuit, this is a backend-specific compilation flag
        semiring='lse-sum' # Use the (R, +, *) semiring, where + is the log-sum-exp and * is the sum
    )

    circuit = ctx.compile(symbolic_circuit).to(device)
    pf_circuit = ctx.integrate(circuit).to(device)
    model = (circuit, pf_circuit)
    train_data, val_data, test_data = load_data(data, binarize = binarize_flag)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    optimizer = torch.optim.SGD([p for p in circuit.parameters() if p.requires_grad == True], lr=lr, momentum=0.95) #torch.optim.Adam([p for p in circuit.parameters() if p.requires_grad == True], lr = lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)

    nll_val, nll_train, model = training(name=name, result_dir=result_dir, max_patience=patience, num_epochs=num_epochs, 
                   model=model, optimizer=optimizer, scheduler=scheduler, training_loader=train_loader, 
                   val_loader=val_loader, device=device, lam=lam, batch_size = batch_size)

    circuit, pf_circuit = model
    test_nll, test_bpd = evaluation(test_loader, device, model_best=model, epoch = 0)
    print(f'Test NLL ={test_nll}, Test BPD = {test_bpd}')
    _, _, aug_test_data = load_data(data, binarize = binarize_flag, augment = True, val = False)
    aug_test_loader = DataLoader(aug_test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    aug_test_val, aug_test_bpd = evaluation(aug_test_loader, device, model_best=model, epoch = 0)

    results_dic = {'nll_val': nll_val.tolist(), 
                    'nll_train': nll_train.tolist(),
                    'test_val': test_nll,
                    'test_bpd': test_bpd,
                    'aug_test_bpd': aug_test_bpd,
                    'aug_test_val': aug_test_val,
                    'KID': 'N/A'}
        
    with open(result_dir + '/' + name + '/results.json', 'w') as json_file:
        json.dump(results_dic, json_file, indent=4)
    print(f"test_bpd = {test_bpd}", f"test_nll = {test_nll}", f"aug_test_bpd = {aug_test_bpd}", f"aug_test_nll = {aug_test_val}")

    #torch.save(circuit, 'models/pc_test_circuit.pt')
    #torch.save(pf_circuit, 'models/pc_test_pf_circuit.pt')
    #circuit = torch.load('models/pc_test_circuit.pt')
    #pf_circuit = torch.load('models/pc_test_pf_circuit.pt')
    #model_best = (circuit, pf_circuit)
    #wandb.log({"test_bpd": test_bpd, "test_loss": test_nll})
    #run.log_artifact(result_dir + '/' + name + '.model')
    #run.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Generative Model.")
    parser.add_argument('--num_input_units', type=int, default=8, help='Number of Input Units (sum, input)')
    parser.add_argument('--num_sum_units', type=int, default=8, help='Number of Input Units (sum, input)')
    parser.add_argument('--height', type=int, default=8, help='Height')
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