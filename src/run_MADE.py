import os
import json
from datetime import datetime
import random
import argparse
import torch
import models
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#import wandb
from train_made import evaluation, training 
from data import load_data
import numpy as np
from util import cross_entropy_loss_fn


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
    n_masks = args.num_masks
    M = args.hidden_layers
    config = {'input_dim': input_dim, 'lr': lr, 'num_epochs': num_epochs, 'max_patience': patience, 'batch_size': batch_size, 'lambda': lam, 'hidden_layers': M}
    #run = wandb.init(entity="rajpal906")#entity="rajpal906", project="MADE", name="unregularized", id="1", config=hyperparameters, settings=wandb.Settings(start_method="fork"))
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparams = f"_E{num_epochs}_BS{batch_size}_LR{lr:.0e}_NM{n_masks}_HL{M}_ID{input_dim}_{data}"
    if args.binarize:
        hyperparams += "_BIN"
    hyperparams += f"_LAM{lam}_PAT{patience}"
    name = f"{current_time}_{hyperparams}"
    if not(os.path.exists(result_dir + '/' + name)):
        os.mkdir(result_dir + '/' + name)

    with open(result_dir + '/' + name + '/config.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)

    model = models.MADE(input_dim=input_dim, hidden_dims=[M], n_masks=n_masks).to(device)
    train_data, val_data, test_data = load_data(data, binarize = binarize_flag)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), drop_last = True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), drop_last = True)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad == True], lr=lr, momentum=0.95) #torch.optim.Adam([p for p in circuit.parameters() if p.requires_grad == True], lr = lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.5)

    nll_val, nll_train, model = training(name=name, result_dir=result_dir, max_patience=patience, num_epochs=num_epochs, 
                   model=model, optimizer=optimizer, scheduler=scheduler, training_loader=train_loader, 
                   val_loader=val_loader, device=device, lam=lam, batch_size = batch_size, loss_fn=cross_entropy_loss_fn)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), drop_last = True)
    test_nll, test_bpd = evaluation(test_loader, loss_fn=cross_entropy_loss_fn, device=device, model_best=model, epoch = 0)
    _, _, aug_test_data = load_data(data, binarize = binarize_flag, augment = True, val = False)
    aug_test_loader = DataLoader(aug_test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), drop_last = True)
    aug_test_val, aug_test_bpd = evaluation(aug_test_loader, loss_fn=cross_entropy_loss_fn, device=device, model_best=model, epoch = 0)

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