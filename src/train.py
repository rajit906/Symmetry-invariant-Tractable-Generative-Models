import torch
import numpy as np
from util import translate_img_batch, translation_configurations
import random

def evaluation(test_loader, name=None, model_best=None, epoch=None):
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
    if model_best is None:
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')

    return loss

def training(name, result_dir, max_patience, num_epochs, model, optimizer, scheduler, training_loader, val_loader, device, lam = 0.):
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

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            batch = batch.to(device)
            loss = model.forward(batch)
            if lam > 0:
                sampled_translations = random.sample(translation_repository, 3)
                s = 0
                for translation in sampled_translations:
                    shift_left, shift_down, shift_right, shift_up = translation
                    translated_img = translate_img_batch(batch, shift_left, shift_down, shift_right, shift_up).to(device)
                    s += torch.abs(loss - model.forward(translated_img)) # L1-Regularization
                loss += loss + lam * s

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        print(f'Epoch: {e}, train nll={loss}, val nll={loss_val}')
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, result_dir + '/' + name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, result_dir + '/' + name + '.model')
                best_nll = loss_val
                patience = 0
                #samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val