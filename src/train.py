import torch
import numpy as np
from util import translate_img_batch, translation_configurations, cross_entropy_loss_fn, bits_per_dim
import random
import wandb

def evaluation(test_loader, device, model_type, loss_fn = cross_entropy_loss_fn, name=None, model_best=None, epoch=None):
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
    if model_type == 'PC':
        if model_best:
            circuit, pf_circuit = model_best
            circuit.eval()
            pf_circuit.eval()

        if model_best is None:
            model_best = (torch.load('models/circuit.pt'), torch.load('models/pf_circuit.pt'))
            circuit, pf_circuit = model_best

        test_lls = 0.0
        log_pf = pf_circuit()
        len_data = 0
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(dim=1) # Add Channel Dimension
            log_output = circuit(batch) # Compute the log output of the circuit
            lls = log_output - log_pf  # Compute the log-likelihood
            test_lls += lls.sum().item()
            len_data += batch.shape[0]
        average_nll = -test_lls / (len_data)
        num_variables = batch.shape[2] #TODO: Keep track if this changes
        bpd = average_nll / (num_variables * np.log(2.0))
        return (average_nll, bpd)
    
    elif model_type == 'MADE':
        if model_best is None:
            model_best = torch.load(name + '.model')
        model_best.eval()
        loss = 0.
        N = 0.
        for _, test_batch in enumerate(test_loader):
            preds = model_best.forward(test_batch)
            loss_t = loss_fn(test_batch, preds)
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]
        input_dim = test_batch.shape[1] # TODO: Keep track of this
        loss = (loss / N)
        bpd = bits_per_dim(loss, input_dim)
        return (loss, bpd)

def training(name, result_dir, model_type, max_patience, num_epochs, model, optimizer, scheduler,
             training_loader, val_loader, device, lam = 0., batch_size = None, loss_fn = None):
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
    bpd_val = []
    nll_train = []
    best_nll = 1000.
    patience = 0
    translation_repository = translation_configurations()
    if model_type == 'PC':
        circuit, pf_circuit = model
    # Main loop
    for e in range(num_epochs):
        # TRAINING
        if model_type == 'PC':
            circuit.train()
            pf_circuit.train()
        elif model_type == 'MADE':
            model.train()
        N = 0.
        train_nll = 0.
        for _, batch in enumerate(training_loader):
            batch = batch.to(device).unsqueeze(dim=1)
            if model_type == 'PC':
                log_output = circuit(batch)
                log_pf = pf_circuit()   
                lls = log_output - log_pf
                loss = -torch.mean(lls)
            elif model_type == 'MADE':
                preds = model.forward(batch)
                loss = loss_fn(batch, preds)
            train_nll += loss
            N = N + batch.shape[0]
            if lam > 0:
                sampled_translations = random.sample(translation_repository, 4)
                s = 0
                for translation in sampled_translations:
                    shift_left, shift_down, shift_right, shift_up = translation
                    translated_batch = translate_img_batch(batch, shift_left, shift_down, shift_right, shift_up).to(device)

                    if model_type == 'PC':
                        translated_batch = translated_batch.unsqueeze(dim=1) # Can this be outside the if?
                        log_translated_output = circuit(translated_batch)
                        log_translated_pf = pf_circuit()
                        translated_lls = log_translated_output - log_translated_pf
                        translated_loss = -torch.mean(translated_lls)

                    elif model_type == 'MADE':
                        translated_preds = model.forward(translated_batch)
                        translated_loss = loss_fn(translated_batch, translated_preds)

                    s += torch.abs(loss - translated_loss)
                loss += lam * s
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        train_nll = train_nll/N
        # Validation
        if model_type == 'PC':
            model = (circuit, pf_circuit)
        loss_val, val_bpd = evaluation(val_loader, device, model_type = model_type, model_best=model, epoch=e)
        print(f'Epoch: {e}, train nll = {train_nll}, val nll = {loss_val}, val bpd = {val_bpd}')
        nll_val.append(loss_val)  # save for plotting
        bpd_val.append(val_bpd)
        nll_train.append(train_nll)

        #wandb.log({"epoch": e,"train_loss": loss,"val_loss": loss_val,"bpd_val": bpd})

        if e == 0:
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                best_nll = loss_val
                patience = 0
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)
    bpd_val = np.asarray(bpd_val)
    nll_train = np.asarray(nll_train)

    return nll_val, bpd_val, nll_train, model