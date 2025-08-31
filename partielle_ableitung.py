import torch
import torch.nn.functional as F
# wird für das partielle Ableiten benötigt
from torch.autograd import grad

# Es geht in dem Programm darum, die Funktion y = w1 * x1 + b zur linearen Regression zu nutzen
# und zu klassifizieren für die Zielwerte 0 und 1 im Sinne von Licht ist an oder aus.
# w1 und b sind die Variablen, nach denen partiell abgeleitet wird.

y = torch.tensor([1.0]) # Zahl 1 als Vektor in einer Dimension
x1 = torch.tensor([1.1]) # Wir berechnen den Gradienten an der Stelle x=1.1 wieder als Vektor
w1 = torch.tensor([2.2], requires_grad=True) # nach w1 wird abgeleitet
b = torch.tensor([0.0], requires_grad=True) # nach b wird abgeleitet

z = w1 * x1 + b
a = torch.sigmoid(z) # Aktivierungsfunktion

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True) # Berechnungsgraph bleibt erhalten
grad_L_b = grad(loss, b, retain_graph=True) # Berechnungsgraph bleibt erhalten

# Verlustgradient manuell ausgeben
print(grad_L_w1)
print(grad_L_b)

# partielle Ableitung berechnen liefert dasselbe Ergebnis
loss.backward()
print(w1.grad)
print(b.grad)




