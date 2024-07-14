import os
import random
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import models
import wandb
import yaml
from merged_scripts.train import evaluation, training 
from data import load_data
import numpy as np
from util import categorical_layer_factory, hadamard_layer_factory, dense_layer_factory, mixing_layer_factory

from Cirkit.cirkit.templates.region_graph import QuadTree
from Cirkit.cirkit.symbolic.circuit import Circuit
from Cirkit.cirkit.pipeline import PipelineContext

random.seed(42)
np.random.seed(42)
os.environ['WANDB_NOTEBOOK_NAME'] = 'hyperparameter_optimization.ipynb'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_dir = 'models'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)
name = 'pc'#Change to regularized

sweep_config = {
    'method': 'grid'
    }
metric = {
'name': 'test_bpd',
'goal': 'minimize'   
}

sweep_config['metric'] = metric

parameters_dict = {
'input_dim': {
    'value': 784
    },
'lam': {
    'values': [0.1, 0.5, 1.0]
    },
'num_epochs': {
    'value': 1
    },
'lr': {
    'values': [1e-1, 1e-2, 1e-3]
    },
'batch_size': {
    'values': [64, 128, 256]
    },
'num_input_units': {
    'value': 8
    },
'num_sum_units': {
    'value': 8
    },
'max_patience': {
    'value': 30 # No patience for now, add momentum?
    },
}


sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="pc_hyperparameter_optimization_chutiya")

def hyperparameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        train_data, val_data, test_data = load_data('mnist', binarize = False)
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True) #, num_workers=os.cpu_count())
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False) #, num_workers=os.cpu_count())
        test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False) #, num_workers=os.cpu_count())
        region_graph = QuadTree(shape=(28, 28))
        symbolic_circuit = Circuit.from_region_graph(region_graph,
                                                    num_input_units=config.num_input_units,
                                                    num_sum_units=config.num_sum_units,
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
        optimizer = torch.optim.SGD([p for p in circuit.parameters() if p.requires_grad == True], lr=config.lr, momentum=0.95)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        _, _, model_best = training(name=name, result_dir=result_dir, max_patience=config.max_patience, num_epochs=config.num_epochs, 
                   model=model, optimizer=optimizer, scheduler=scheduler, training_loader=train_loader, 
                   val_loader=val_loader, device=device, lam=config.lam, batch_size = config.batch_size)
        test_nll, test_bpd = evaluation(test_loader, device, model_best=model_best)
        wandb.log({"test_bpd": test_bpd, "test_bpd": test_nll})
        circuit, pf_circuit = model
        torch.save(circuit, 'models/circuit.pt')
        torch.save(pf_circuit, 'models/pf_circuit.pt')
        wandb.log_artifact('models/circuit.pt')
        wandb.log_artifact('models/pf_circuit.pt')

wandb.agent(sweep_id, hyperparameter_sweep)