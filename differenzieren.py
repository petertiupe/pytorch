import torch
# Die Differenz-Maschine von Python wird auch als ### Autograd ### bezeichnet.
# Basis für Pytorchs Autograd sind sogenannte Berechnungsgraphen, die jeden einzelnen Schritt
# einer Berechnung festhalten und dann automatisch via Kettenregel die Ableitung bestimmen können.
# Tensor mit requires_grad=True erstellen, sorgt dafür, dass der Berechnungsgraph erzeugt wird.
# Dabei wird der Berechnungsgraph dadurch erzeugt, dass sich Autograd alle Berechnungen
# merkt, die auf einem Tensor durchgeführt wurden.
# Mathematisch ausgedrückt muss requires_grad für jede Variable gesetzt werden, für die eine
# partielle Ableitung erstellt werden muss.
x = torch.tensor(2.0, requires_grad=True)

# Funktion definieren
y = 5 * x**3 + 52 * x**2 + 3 * x - 2

# Ableitung berechnen
# backward nimmt einem alles bzgl. der Differentiation von Funktionen ab.
y.backward()

# Ergebnis ausgeben
print("Der numerische Wert der Ableitung an x=2.0 ist:", x.grad) # Ergebnis ist 271, habe ich nachgerechnet...