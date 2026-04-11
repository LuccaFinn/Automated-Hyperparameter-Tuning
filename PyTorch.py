import torch
from torch import nn
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

# --- 1. Daten laden (ACHTUNG: PyTorch braucht spezielle Datentypen!) ---
data = load_iris()
# PyTorch verlangt für Input-Daten zwingend 'float32' (Kommastellen)
X = data.data.astype(np.float32) 
# PyTorch verlangt für Klassen-Labels zwingend 'int64' (Ganze Zahlen)
y = data.target.astype(np.int64) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Das PyTorch-Modell bauen ---
# Wir bauen eine Klasse, die von nn.Module erbt. 
# WICHTIG: Die Parameter, die wir tunen wollen, müssen ins __init__!
class VariablePyTorchNet(nn.Module):
    def __init__(self, hidden_units=10, activation_fn=nn.ReLU()):
        super().__init__()
        # Layer 1: 4 Features (Iris) gehen rein, 'hidden_units' kommen raus
        self.dense0 = nn.Linear(4, hidden_units) 
        # Die dynamische Aktivierungsfunktion
        self.activation = activation_fn
        # Layer 2 (Output): 'hidden_units' gehen rein, 3 Klassen gehen raus
        self.output = nn.Linear(hidden_units, 3) 

    def forward(self, X):
        X = self.dense0(X)
        X = self.activation(X)
        X = self.output(X)
        return X

# --- 3. Die Brücke: Skorch ---
# Wir verpacken das PyTorch-Modell, damit scikit-learn damit arbeiten kann.
net = NeuralNetClassifier(
    VariablePyTorchNet,
    max_epochs=50,
    lr=0.1, # Standard-Lernrate
    iterator_train__shuffle=True, # Mischt die Daten beim Training (PyTorch Standard)
    verbose=0 # Schaltet den Output ab, damit die Konsole nicht überläuft
)

# --- 4. Den Suchraum definieren ---
# WICHTIGER UNTERSCHIED: Wenn du Parameter deines eigenen PyTorch-Modells
# (VariablePyTorchNet) ändern willst, musst du in Skorch "module__" davorschreiben!
param_grid = {
    'lr': [0.01, 0.05, 0.1], # Die Lernrate (Schrittgröße) des Optimizers
    'module__hidden_units': [10, 20, 30], # Unser selbst definierter Parameter
    # Hier nutzen wir nun die ECHTEN mathematischen PyTorch-Funktionen!
    'module__activation_fn': [nn.ReLU(), nn.Tanh(), nn.ELU()] 
}

# --- 5. Den Loop starten ---
grid_search = GridSearchCV(
    estimator=net, 
    param_grid=param_grid, 
    cv=3, # Auf 3 reduziert, damit es schneller geht
    scoring='accuracy',
    n_jobs=-1
)

print("Starte PyTorch Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# --- Ergebnisse auswerten ---
print("\n--- ERGEBNISSE ---")
print(f"Beste Parameter-Kombination: {grid_search.best_params_}")
print(f"Beste Genauigkeit (Cross-Validation): {grid_search.best_score_:.4f}")