print('..running')
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_model_summary import summary
import yaml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from util import samples_generated, samples_real, plot_curve
import idf
from train import evaluation, training 
from data import load_data
from neural_networks import nnetts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1000
train_data, val_data, test_data = load_data('mnist')
# Create data loaders
training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

result_dir = 'results/exp_test'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)
name = 'idf-4'

D = 784   # input dimension
M = D  # the number of neurons in scale (s) and translation (t) nets
lr = 1e-3 # learning rate
num_epochs = 100 # max. number of epochs
max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
num_flows = 4 # The number of invertible transformations

hyperparameters = {'D': D, 
                   'M': M,
                   'lr': lr,
                   'num_epochs': num_epochs,
                   'max_patience': max_patience,
                   'num_flows': num_flows,
                   'batch_size': batch_size
                    }

with open(result_dir + '/hyperparameters.yaml', 'w') as file:
    yaml.dump(hyperparameters, file)

netts = nnetts(D, M)
model = idf.IDF4(netts, num_flows, D=D).to(device)
#print(summary(model, torch.zeros(1, 64), show_input=False, show_hierarchical=False))
optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
# Training procedure
nll_val = training(name=name, result_dir = result_dir, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader, device=device)

with open(result_dir + '/train_loss.txt', "w") as file:
    for item in nll_val:
        file.write(f"{item}\n")

test_loss = evaluation(name=result_dir + '/' + name, test_loader=test_loader)
f = open(result_dir + '/test_loss.txt', "w")
f.write(str(test_loss))
f.close()

samples_generated(result_dir + '/' + name, test_loader, 28)
plot_curve(result_dir + '/' + name, nll_val)