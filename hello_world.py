#!/usr/bin/python3
import torch
# So kann man testen, ob die GPU bei Berechnungen auf dem Mac benutzt werden kann. 
# Bei mir ist das Ergebnis true 
print(torch.backends.mps.is_available())
print(torch.__version__)
print("Diese Zeile wird ausgegeben!")