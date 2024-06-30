import torch
import numpy as np
from util import translate_img_batch, translation_configurations
import random
import wandb

def evaluation(test_loader, device, name=None, model_best=None, epoch=None):
    """
    Evaluates the model on the test dataset and computes the negative log-likelihood loss.

    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        name (str, optional): The name of the saved model file to load if `model_best` is not provided.
        model_best (torch.nn.Module, optional): The model to be evaluated. If None, the model will be loaded from file.
        epoch (int, optional): The current epoch number. If None, the function prints the final loss.

    Returns:
        float: The computed negative log-likelihood loss on the test dataset.
    """
    if model_best:
        circuit, pf_circuit = model_best
        circuit.eval()
        pf_circuit.eval()

    if model_best is None:
        model_best = torch.load(name + '.model')

    test_lls = 0.0
    log_pf = pf_circuit()

    for i, batch in enumerate(test_loader):
        batch = batch.to(device).unsqueeze(dim=1)   # Add a channel dimension
        log_output = circuit(batch)                 # Compute the log output of the circuit
        lls = log_output - log_pf                   # Compute the log-likelihood
        test_lls += lls.sum().item()

    average_nll = -test_lls / ((i+1) * batch.shape[0])
    bpd = average_nll / (batch.shape[2] * np.log(2.0))
    print(f"Average test LL: {average_nll:.3f}") #TODO: log to wandb
    print(f"Bits per dimension: {bpd}") #TODO: log to wandb
    if epoch is None:
        print(f'FINAL LOSS: nll={average_nll}')

    return average_nll

def training(name, result_dir, max_patience, num_epochs, model, optimizer, scheduler,
             training_loader, val_loader, device, lam = 0., batch_size = None):
    """
    Trains a given model using the specified training and validation data loaders.

    This function includes an option to impose translation invariance by augmenting the training data with translations 
    and adding a corresponding penalty term to the loss function.

    Args:
        name (str): The name to use for saving the trained model.
        result_dir (str): The directory where the model will be saved.
        max_patience (int): The maximum number of epochs to wait for improvement in validation loss before stopping.
        num_epochs (int): The number of epochs to train the model.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        training_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to train the model on (e.g., 'cpu' or 'cuda').
        lam (float): The regularization parameter for translation invariance (default is 0).

    Returns:
        np.ndarray: Array of negative log-likelihoods (validation losses) of the model.
    """
    nll_val = []
    best_nll = 1000.
    patience = 0
    translation_repository = translation_configurations()
    circuit, pf_circuit = model
    # Main loop
    for e in range(num_epochs):
        # TRAINING
        circuit.train()
        pf_circuit.train()
        for _, batch in enumerate(training_loader):
            batch = batch.to(device).unsqueeze(dim=1)
            log_output = circuit(batch)
            log_pf = pf_circuit()   
            lls = log_output - log_pf
            loss = -torch.mean(lls)
            if lam > 0:
                sampled_translations = random.sample(translation_repository, 4)
                s = 0
                for translation in sampled_translations:
                    shift_left, shift_down, shift_right, shift_up = translation
                    translated_batch = translate_img_batch(batch, shift_left, shift_down, shift_right, shift_up).to(device)
                    translated_batch = translated_batch.unsqueeze(dim=1)
                    log_translated_output = circuit(translated_batch)
                    log_translated_pf = pf_circuit()
                    translated_lls = log_translated_output - log_translated_pf
                    translated_loss = -torch.mean(translated_lls)
                    s += torch.abs(loss - translated_loss)
                loss += loss + lam * s
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        # Validation
        model = (circuit, pf_circuit)
        loss_val = evaluation(val_loader, device, model_best=model, epoch=e)
        print(f'Epoch: {e}, train nll={loss}, val nll={loss_val}')
        nll_val.append(loss_val)  # save for plotting

        #wandb.log(
            #{
                #"epoch": e,
                #"train_loss": loss,
                #"val_loss": loss_val,
            #}
        #)

        if e == 0:
            print('saved!')
            #TODO: Figure out how to save model parameters.
            #torch.save(circuit, result_dir + '/' + name + '.pt')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                #torch.save(circuit, result_dir + '/' + name + '.pt')
                best_nll = loss_val
                patience = 0
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val