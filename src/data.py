from sklearn import datasets
from sklearn.datasets import load_digits
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import numpy as np

class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    
# Custom dataset class to return only images
class MNISTWithoutLabels(datasets.MNIST):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)  # Ignore the label
        return img

def load_data(name):
    if name == 'sklearn':
        train_data = Digits(mode='train')
        val_data = Digits(mode='val')
        test_data = Digits(mode='test')
    if name == 'mnist':

        transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))])

        train_data = MNISTWithoutLabels(root='./data', train=True, download=True, transform=transform)
        test_data = MNISTWithoutLabels(root='./data', train=False, download=True, transform=transform)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
    return (train_data, val_data, test_data)