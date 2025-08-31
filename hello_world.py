#!/usr/bin/python3
import sys
import torch
import torchvision

print("Python-Version: ", sys.version)

# So kann man testen, ob die GPU bei Berechnungen auf dem Mac benutzt werden kann. 
# Bei mir ist das Ergebnis true 
print(torch.backends.mps.is_available())

print("torch-Version: ", torch.__version__)


print("torchvision-Version: ", torchvision.__version__)


x = torch.rand(5, 3)
print("Zufällige 5x3-Matrix:\n", x)