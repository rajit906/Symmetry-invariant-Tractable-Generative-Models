print('running')
import os
import torch
from torch.utils.data import DataLoader
import pickle as pkl
from joblib import dump
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from util import translate_img_batch, cross_entropy_loss_fn
from data import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_data = load_data('mnist', binarize=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

result_dir = 'results_made/exp_1'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)
name = 'made'
model = torch.load(result_dir + '/' + name + '.model').to(device)
model.eval()

translations = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)] #, (1,1,0,0), (0,0,1,1), (2,1,0,0), (1,2,0,0), (0,0,1,2), (0,0,2,1), (2,2,0,0), (0,0,2,2)]
lls_translation = {}
for translation in translations:
    print(translation)
    shift_left, shift_down, shift_right, shift_up = translation
    lls = []
    for img in test_loader:
        translated_img = translate_img_batch(img, shift_left, shift_down, shift_right, shift_up)
        preds = model.forward(img)
        translated_preds = model.forward(translated_img)
        ll = -(cross_entropy_loss_fn(img, preds) - cross_entropy_loss_fn(translated_img, translated_preds))
        lls.append(ll)
    lls_translation[translation] = lls

with open(result_dir + '/translation_lls_1.pkl', 'wb') as file:
    dump(lls_translation, file)
