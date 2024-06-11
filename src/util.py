# Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
# Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743
# Source: https://github.com/jornpeters/integer_discrete_flows

import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np

class RoundStraightThrough(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def log_min_exp(a, b, epsilon=1e-8):
    """
    Source: https://github.com/jornpeters/integer_discrete_flows
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
     log(exp(a) - exp(b))
     c + log(exp(a-c) - exp(b-c))
     a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
        F.logsigmoid((x + 0.5 - mean) / scale),
        F.logsigmoid((x - 0.5 - mean) / scale))

    return logp

def samples_real(name, test_loader, D):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (D, D))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, D, extra_name=''):
    x = next(iter(data_loader)).detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (D, D))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()

def translate_img_batch(img_batch, shift_left = 0, shift_down = 0, shift_right = 0, shift_up = 0):
    batch_size = img_batch.size(0)
    old_img_batch = img_batch.view(batch_size, 1, 28, 28).clone()
    new_img_batch = torch.zeros_like(old_img_batch)
    if shift_left > 0 and shift_down > 0:
        new_img_batch[:, 0, shift_down:, :-shift_left] = old_img_batch[:, 0, :-shift_down, shift_left:]
    elif shift_left > 0:
        new_img_batch[:, 0, :, :-shift_left] = old_img_batch[:, 0, :, shift_left:]
    elif shift_down > 0:
        new_img_batch[:, 0, shift_down:, :] = old_img_batch[:, 0, :-shift_down, :]
    elif shift_right > 0 and shift_up > 0:
        new_img_batch[:, :, :-shift_up, shift_right:] = old_img_batch[:, :, shift_up:, :-shift_right]
    elif shift_right > 0:
        new_img_batch[:, :, :, :-shift_right] = old_img_batch[:, :, :, shift_right:]
    elif shift_up > 0:
        new_img_batch[:, :, :-shift_up, :] = old_img_batch[:, :, shift_up:, :]
    else:
        new_img_batch = old_img_batch

    new_img_batch_flat = new_img_batch.view(batch_size, 784)
    return new_img_batch_flat

def visualize_translated_images(original_images, translated_images):
    original_images = original_images.view(-1, 28, 28)
    translated_images = translated_images.view(-1, 28, 28)
    batch_size = original_images.size(0)
    fig, axes = plt.subplots(2, batch_size, figsize=(15, 5))
    for i in range(batch_size):
        axes[0, i].imshow(original_images[i].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i+1}')
        
        axes[1, i].imshow(translated_images[i].cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Translated {i+1}')
    plt.tight_layout()
    plt.show()

def translation_configurations():
    return [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,1,0,0), (0,0,1,1), 
            (2,1,0,0), (1,2,0,0), (0,0,1,2), (0,0,2,1), (2,2,0,0), (0,0,2,2),
            (2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2), (3,0,0,0), (0,3,0,0), 
            (0,0,3,0), (0,0,0,3), (3,3,0,0), (0,0,3,3), (3,2,0,0), (0,0,3,2),
            (3,1,0,0), (0,0,3,1), (1,3,0,0), (0,0,1,3), (4,0,0,0), (0,4,0,0),
            (0,0,4,0), (0,0,0,4), (4,1,0,0), (1,4,0,0), (4,2,0,0), (2,4,0,0),
            (4,3,0,0), (3,4,0,0), (4,4,0,0), (4,4,0,0), (0,0,4,1), (0,0,1,4),
            (0,0,4,2), (0,0,2,4), (0,0,4,3), (0,0,3,4), (0,0,4,4), (0,0,4,4)]