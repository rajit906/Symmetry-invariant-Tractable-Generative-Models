print('running')
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from pytorch_model_summary import summary
import matplotlib.pyplot as plt
import yaml
import pickle as pkl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from util import samples_generated, samples_real, plot_curve, translate_img_batch, visualize_translated_images
import idf
from train import evaluation, training 
from data import load_data
from neural_networks import nnetts

_, _, test_data = load_data('mnist')
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

result_dir = 'results/exp_1'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)
name = 'best_idf-4'
model = torch.load(result_dir + '/' + name + '.model')

translations = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,1,0,0), (0,0,1,1), (2,1,0,0), (1,2,0,0), (0,0,1,2), (0,0,2,1), (2,2,0,0), (0,0,2,2)]
lls_translation = {}
for translation in translations:
    print(translation)
    shift_left, shift_down, shift_right, shift_up = translation
    lls = []
    for img in test_loader:
        translated_img = translate_img_batch(img, shift_left, shift_down, shift_right, shift_up)
        ll = -(model.forward(img) - model.forward(translated_img))
        lls.append(ll)
    lls_translation[translation] = lls

with open(result_dir + '/translation_lls.pkl', 'wb') as file:
    pkl.dump(lls_translation, file)
