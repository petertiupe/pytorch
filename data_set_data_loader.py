# PyTorch hat zwei Möglichkeiten, um Daten zu laden
#   - torch.utils.data.DataLoader und 
#   - torch.utils.data.Dataset. 

# Dataset speichert die Daten und ihre zugehörigen Labels
# DataLoader umgibt Dataset mit einem Iterable, so dass man die Daten
# nacheinander laden kann und nicht alle gleichzeitig verarbeiten muss.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
# Modified National Institute of Standards and Technology, dafür steht MNIST
test_data = datasets.FashionMNIST(    
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(training_data)
print(test_data)

batch_size = 64

# Hier werden die Daten nun einem DataLoader übergeben
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

print( torch.accelerator.is_available() )

