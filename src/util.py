# Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
# Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743
# Source: https://github.com/jornpeters/integer_discrete_flows

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn import metrics
import torchmetrics
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
from torchvision import transforms


### Integer Discrete Flows

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


def log_discretized_logistic(x, mean, logscale, inverse_bin_width = 1.):
    scale = torch.exp(logscale)

    logp = log_min_exp(
        F.logsigmoid((x + 0.5 / inverse_bin_width - mean) / scale),
        F.logsigmoid((x - 0.5 / inverse_bin_width - mean) / scale))
    return logp


def log_mixture_discretized_logistic(x, mean, logscale, pi, inverse_bin_width = 1.):
    scale = torch.exp(logscale)
    
    p = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale) \
        - torch.sigmoid((x - 0.5 / inverse_bin_width - mean) / scale)
    p = torch.sum(p * pi, dim=-1)
    logp = torch.log(p + 1e-8)

    return logp

### Masked Autoencoder for Density Estimation

def cross_entropy_loss_fn(x, preds):
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
    return loss.sum(dim=1).mean()

### Probabilistic Circuits

from Cirkits.cirkit.symbolic.parameters import LogSoftmaxParameter, SoftmaxParameter, ExpParameter, Parameter, TensorParameter
from Cirkits.cirkit.symbolic.layers import CategoricalLayer, DenseLayer, HadamardLayer, MixingLayer
from Cirkits.cirkit.symbolic.initializers import NormalInitializer
from Cirkits.cirkit.utils.scope import Scope

def categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int
) -> CategoricalLayer:
    return CategoricalLayer(
        scope, num_units, num_channels, num_categories=256,
        logits_factory=lambda shape: Parameter.from_unary(
            LogSoftmaxParameter(shape),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-2))
        )
    )

def hadamard_layer_factory(
    scope: Scope, num_input_units: int, arity: int
) -> HadamardLayer:
    return HadamardLayer(scope, num_input_units, arity)

def dense_layer_factory(
    scope: Scope,
    num_input_units: int,
    num_output_units: int
) -> DenseLayer:
    return DenseLayer(
        scope, num_input_units, num_output_units,
        weight_factory=lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1))
        )
    )

def mixing_layer_factory(
    scope: Scope, num_units: int, arity: int
) -> MixingLayer:
    return MixingLayer(
        scope, num_units, arity,
        weight_factory=lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1e-1))
        )
    )


### General Purpose


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

def translation_configurations(n = 4):
    #{1: [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,1,0,0), (0,0,1,1), (1,0,0,1), (0,1,1,0)],
    # 2: [(2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2), (2,2,0,0), (0,0,2,2), (2,0,0,2), (0,2,2,0)],
    # 3: [(3,0,0,0), (0,3,0,0), (0,0,3,0), (0,0,0,3), (3,3,0,0), (0,0,3,3), (3,0,0,3), (0,3,3,0)],
    # 4: [(4,0,0,0), (0,4,0,0), (0,0,4,0), (0,0,0,4), (4,4,0,0), (0,0,4,4), (4,0,0,4), (0,4,4,0)]}
    return [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,1,0,0), (0,0,1,1), 
            (2,1,0,0), (1,2,0,0), (0,0,1,2), (0,0,2,1), (2,2,0,0), (0,0,2,2),
            (2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2), (3,0,0,0), (0,3,0,0), 
            (0,0,3,0), (0,0,0,3), (3,3,0,0), (0,0,3,3), (3,2,0,0), (0,0,3,2),
            (3,1,0,0), (0,0,3,1), (1,3,0,0), (0,0,1,3), (4,0,0,0), (0,4,0,0),
            (0,0,4,0), (0,0,0,4), (4,1,0,0), (1,4,0,0), (4,2,0,0), (2,4,0,0),
            (4,3,0,0), (3,4,0,0), (4,4,0,0), (4,4,0,0), (0,0,4,1), (0,0,1,4),
            (0,0,4,2), (0,0,2,4), (0,0,4,3), (0,0,3,4), (0,0,4,4), (0,0,4,4),
            (0,1,1,0), (0,1,2,0), (0,1,3,0), (0,1,4,0), (0,2,1,0), (0,3,1,0),
            (0,4,1,0), (0,2,2,0), (0,3,3,0), (0,4,4,0), (0,3,2,0), (0,2,3,0),
            (0,3,4,0), (0,4,3,0), (1,0,0,1), (1,0,0,2), (1,0,0,3), (1,0,0,4),
            (2,0,0,1), (3,0,0,1), (4,0,0,1), (2,0,0,2), (2,0,0,3), (2,0,0,4),
            (3,0,0,2), (3,0,0,3), (3,0,0,4), (4,0,0,2), (4,0,0,3), (4,0,0,4)]

def bits_per_dim(nll_val, dim):
    return (nll_val / dim) / np.log(2)

def update_imports_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expressions to match and replace import statements
    updated_content = re.sub(r'from cirkit\.', 'from Cirkits.cirkit.', content)
    updated_content = re.sub(r'import cirkit\.', 'import Cirkits.cirkit.', updated_content)

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_imports_in_file(file_path)


def plot_samples(model, n, model_type, save_dir = None):
    # Ensure n is a perfect square for an n x n grid
    grid_size = int(n ** 0.5)
    assert grid_size ** 2 == n, "n must be a perfect square for an n x n grid"

    # Sample data from the model
    x = sample_model(model, n, model_type = model_type)
    dim = x.shape[1] * x.shape[2] * x.shape[3]
    samples = [(x[i].reshape(1,-1), 0) for i in range(len(x))]
    data_loader = DataLoader(samples,  batch_size=1)
    nlls = compute_nlls(model, data_loader=data_loader, model_type = model_type).tolist()
    samples = [x[i].reshape(int(dim**0.5),int(dim**0.5)).detach().numpy() for i in range(len(x))]
    samples_nll = list(zip(samples,nlls))
    
    # Sort images by nll in increasing order
    samples_nll.sort(key=lambda pair: pair[1])

    # Close any previous figures to prevent multiple grids from being plotted
    plt.close('all')

    # Create subplots
    fig1, axes1 = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    # Loop through each sample and plot it
    for i, (img, nll) in enumerate(samples_nll):
        ax = axes1[i // grid_size, i % grid_size]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{round(nll, 1)}", fontsize=20, fontweight='bold')
        ax.axis('off')

    # Adjust layout and show plot
    fig1.tight_layout()
    #plt.show()
    if save_dir:
        plt.savefig(save_dir, format='pdf', bbox_inches='tight')
    plt.close(fig1)
    
    return fig1

def sample_model(model, n, model_type):
    if model_type == 'MADE':
        x = model.sample(n)
    if model_type == 'PC':
        circuit, _ = model
        x = circuit.sample_forward(n)[0]
        h = int(x.shape[-1]**0.5)
        w = h
        x = x.reshape(n,1,h,w)
    return x

def compute_nlls(model, data_loader, model_type = 'MADE'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlls = []
    if model_type == 'MADE':
        model = model.to(device)
        for _, (batch, _) in enumerate(data_loader):
            batch = batch.to(device)
            preds = model.forward(batch)
            nll = cross_entropy_loss_fn(batch, preds)
            nlls.append(nll.item())
    elif model_type == 'PC':
        circuit, pf_circuit = model
        circuit = circuit.to(device)
        pf_circuit = pf_circuit.to(device)
        for _, (batch, _) in enumerate(data_loader):
            batch = batch.to(device)
            if len(batch.shape) == 2:
                    batch = batch.unsqueeze(dim=1)
            log_output = circuit(batch)
            log_pf = pf_circuit()
            loss = -torch.mean(log_output - log_pf)
            nlls.append(loss.item())
    return np.array(nlls)


def roc_pc(test_loader, test_loader_ood, model, model_type, nll_mnist = None, nll_ood = None):
    if not (nll_mnist.tolist() and nll_ood.tolist()):
        nll_mnist = compute_nlls(model, test_loader, model_type=model_type)
        nll_ood = compute_nlls(model, test_loader_ood, model_type=model_type)
    nll_ood = nll_ood[:len(nll_mnist)]
    max_nll_mnist = max(nll_mnist)
    max_nll_ood = max(nll_ood)
    nlls_mnist = [(nll/max_nll_mnist, 1) for nll in nll_mnist]
    nlls_ood = [(nll/max_nll_ood, 0) for nll in nll_ood]
    nlls = nlls_mnist + nlls_ood
    nlls = sorted(nlls, key=lambda x: x[0], reverse = True)
    nll_scores = [val[0] for val in nlls]
    nll_labels = [val[1] for val in nlls]
    fpr, tpr, thresholds = metrics.roc_curve(nll_labels, nll_scores, pos_label=1)
    roc_auc = metrics.roc_auc_score(nll_labels, nll_scores)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(nll_labels, nll_scores)
    pr_auc = metrics.auc(recall, precision)
    return fpr, tpr, thresholds, roc_auc, precision, recall, pr_thresholds, pr_auc, nll_mnist, nll_ood
# Specify the root directory to start the search
#root_directory = 'Cirkits/'  # Change this to your target directory

#process_directory(root_directory)

def compute_KID(true, samples, subset_size, device, feature = 2048):
    metric = KernelInceptionDistance(feature=feature, subset_size=subset_size, normalize=True,
                                    gamma = None, coef = 1.0, degree = 3).to(device)
    metric.update(true, real=True)
    metric.update(samples, real=False)
    mean, std = metric.compute()
    return mean.item(), std.item()

def preprocess_single_image(image, preprocess):
    image = transforms.ToPILImage()(image.squeeze(0))
    image = image.convert("RGB")
    image = preprocess(image)
    return image

def preprocess_samples(samples, preprocess):
    return torch.stack([preprocess_single_image(img, preprocess) for img in samples])

def extract_features(data, model):
    all_features = []
    with torch.no_grad():
        for images in data:
            images = images.unsqueeze(0)
            features = model(images)  # Output: [batch_size, 2048] from the removed classification head
            all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)

def OOD_detection(model, data_loader, nll_trained_model, epsilon, model_type):
    nlls = compute_nlls(model, data_loader, model_type = model_type)
    ood_count = 0.0
    count = 0.0

    for i in range(len(nlls)):
        nll = nlls[i]
        if np.abs(nll - nll_trained_model) > epsilon:
            ood_count += 1.0
        count += 1.0

    ood_fraction = ood_count / count
    id_fraction = 1.0 - ood_fraction

    return ood_fraction, id_fraction

def get_epsilon(model, data, nll_trained_model, model_type, K, M, alpha=0.99):
    """
    K: number of bootstrap sampled data sets
    M: size of each K data sets
    alpha: confidence level
    """
    epsilons = []
    indices = np.arange(len(data))
    for k in range(K):
        np.random.shuffle(indices)
        subset_indices = indices[:M]
        sampler = SubsetRandomSampler(subset_indices)
        data_loader = DataLoader(data, batch_size=M, sampler=sampler)
        e_k = np.abs(np.mean(compute_nlls(model, data_loader, model_type) - nll_trained_model))
        epsilons.append(e_k)

    epsilon_M_alpha = np.quantile(epsilons, alpha)

    return epsilons, epsilon_M_alpha

def typicality_test(model, train_data, val_data, test_data, test_data_ood, K, alpha, model_type, M = 2):
        train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
        nll_trained_model = np.mean(compute_nlls(model, train_loader, model_type = model_type))
        _, epsilon = get_epsilon(model, val_data, nll_trained_model, model_type, K=K, M=M, alpha=alpha)
        test_loader = DataLoader(test_data, batch_size=M, shuffle=False)
        test_loader_ood = DataLoader(test_data_ood, batch_size=M, shuffle=False)
        ood_mnist, _ = OOD_detection(model, test_loader, nll_trained_model, epsilon, model_type)
        ood_data, _ = OOD_detection(model, test_loader_ood, nll_trained_model, epsilon, model_type)
        return ood_mnist, ood_data
