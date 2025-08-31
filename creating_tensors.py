#!/usr/bin/python3
import torch

# So werden Tensoren mit dem entsprechenden Rang erzeugt. Der Rang einer Matrix hat nichts mit
# der Dimension von zugehörigen Vektoren zu tun.
tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1,2,3]) # Vektordimension ist 3 Rang des Tensors ist 1
tensor2d = torch.tensor([[1,2,3], [4,5,6]]) # 2x3 - Matrix vom Rang 2

print(tensor0d.shape)
print(tensor1d.shape)
print(tensor2d.shape)

tensor2dT = tensor2d.T # transformiert die Matrix
print(tensor2dT)

# Matrizenmultiplikation
matrix_multiplication = tensor2d.matmul(tensor2dT)
print(matrix_multiplication)

# Matrizenmultiplikation kann man auch mit einem @ schreiben
matrix_with_at = tensor2d @ tensor2dT
print(matrix_with_at)

# Umformen einer Matrix
viewed = tensor2d.view(1,6)
print(viewed)

# genauso geht es auch mit der Funktion aus numpy:
# Der einzige Unterschied besteht darin, dass reshape Vektoren ergänzt, während view
# abbricht, wenn die neue Dimension nicht passt.
reshaped = tensor2d.reshape(1,6)
print(reshaped)