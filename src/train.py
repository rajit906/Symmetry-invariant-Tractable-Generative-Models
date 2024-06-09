import torch
import numpy as np
from util import translate_img_batch
import random

def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
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
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss

def training(name, result_dir, max_patience, num_epochs, model, optimizer, training_loader, val_loader, lam = 0.):
    nll_val = []
    best_nll = 1000.
    patience = 0
    translation_repository = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,1,0,0), (0,0,1,1), 
                              (2,1,0,0), (1,2,0,0), (0,0,1,2), (0,0,2,1), (2,2,0,0), (0,0,2,2),
                              (2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2), (3,0,0,0), (0,3,0,0), 
                              (0,0,3,0), (0,0,0,3), (3,3,0,0), (0,0,3,3), (3,2,0,0), (0,0,3,2),
                              (3,1,0,0), (0,0,3,1), (1,3,0,0), (0,0,1,3), (4,0,0,0), (0,4,0,0),
                              (0,0,4,0), (0,0,0,4), (4,1,0,0), (1,4,0,0), (4,2,0,0), (2,4,0,0),
                              (4,3,0,0), (3,4,0,0), (4,4,0,0), (4,4,0,0), (0,0,4,1), (0,0,1,4),
                              (0,0,4,2), (0,0,2,4), (0,0,4,3), (0,0,3,4), (0,0,4,4), (0,0,4,4)]

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            
            loss = model.forward(batch)
            if lam > 0:
                sampled_translations = random.sample(translation_repository, 5)
                s = 0
                for translation in sampled_translations:
                    shift_left, shift_down, shift_right, shift_up = translation
                    translated_img = translate_img_batch(batch, shift_left, shift_down, shift_right, shift_up)
                    s += (loss - model.forward(translated_img))**2
                loss += loss + lam * s**2 

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
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