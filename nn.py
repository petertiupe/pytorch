import torch
# Um die Netzarchitektur zu implementiern, leitet man von der Klasse torch.nn.Module ab.
# In Python leitet man von einer Klasse ab, indem man Folgedes schreibt:
# Man erbt in Python von einer Klasse, indem man sie in der Definition der Subklasse
# in Klammern hinzufügt

class NeuralNetwork(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs):
        """Legt das Layout des Neuronalen Netzwerks im Konstruktor fest.
        Args:
            num_inputs: Anzahl der Eingangsknoten im Netz
            num_outputs: Anzahl der Ausgangsknotn im Netz

        """
        super().__init__()

        self.layers = torch.nn.Sequential(
            # erster versteckter Layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(), # nicht lineare Aktivierungsfunktion
            # zweiter verstecker Layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(), # nichtlineare Aktivierungsfunktion

            # Ausgabe-Layer
            torch.nn.Linear(20, num_outputs)
        )

        def forward(self, x):
            # Sequential hat eine __call__-Methode implementiert, die wird in dem folgenden
            # Aufruf gecalled.
            logits = self.layers(x)
            return logits

# Verwendung der Klasse und Ausgabe des Modells
model = NeuralNetwork(50,3)
print(model)

# Anzahl der trainierbaren Parameter ausgeben
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Anzahl der trainierbaren Model-Paramter: ", num_params)

# Zugriff auf die Gewichte der ersten Schicht:
print(model.layers[0].weight)
print(model.layers[0].weight.shape)

# Zugriff auf die Bias-Werte
print(model.layers[0].bias)
print(model.layers[0].bias.shape)

