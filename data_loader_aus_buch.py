import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]    
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])

y_test = torch.tensor([0,1])



# Die Dataset-Klasse ist die Klasse, die beschreibt, wie die Testdaten aussehen.
# Im Wesentlichen ist __getitem__ und __len__ zu implementieren.
class ToyDataset(Dataset): # ToyDataset erbt von Dataset
    # Konstruktor der Klasse
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y # Python kann Tupel einfach als kommaseparierte Liste zurückgeben.
    
    def __len__(self):
        return self.labels.shape[0]

# Mit den oben definierten Testdaten werden die zwei ToyDatasets erzeugt:
train_ds = ToyDataset(X_train, y_train)
test_ds  = ToyDataset(X_test, y_test)

# Die Dataset-Klasse kann nun genutzt werden, um einen DataLoader zu nutzen.

# stellt die Reproduzierbarkeit sicher, indem die Zufallswerte vom Zufallszahlengenerator von PyTorch jedes
# Mal gleich initialisiert werden. Ohne diese Zeile wären die Ergebnisse je Lauf ein wenig abweichend.
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True, # mischen ja, braucht man, damit beim Training das Modell nicht in sich selbst hängen bleibt.
    num_workers=0, # Anzahl der Hintergrundprozesse zur Parallelisierung
    # drop_last=True # sorgt dafür, dass der letzte kleinere Datensatz weggeworfen wird
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# Iteration über Trainings-Data-Loader+
for idx, (x,y) in enumerate(train_loader): # enumerate liefert hier bereits ein Tupel aus dem Index und dem Tupel aus (x,y) daher die elegante Schreibweise
    print(f"Batch {idx+1}:", x, y)

# Iteration über die Testdaten
for idx, (x,y) in enumerate(test_loader):
    print("Batch" , idx+1 , x, y)
