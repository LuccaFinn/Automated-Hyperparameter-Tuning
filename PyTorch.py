import torch
from torch import nn
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

#  Daten einladen
data = load_iris()
# Input-Datentyp festlegen
X = data.data.astype(np.float32) 
# Datentyp für Klasenlabels festlegen
y = data.target.astype(np.int64) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class VariablePyTorchNet(nn.Module):
    def __init__(self, hidden_units=10, activation_fn=nn.ReLU()):
        super().__init__()

        self.dense0 = nn.Linear(4, hidden_units) 

        self.activation = activation_fn

        self.output = nn.Linear(hidden_units, 3) 

    def forward(self, X):
        X = self.dense0(X)
        X = self.activation(X)
        X = self.output(X)
        return X

net = NeuralNetClassifier(
    VariablePyTorchNet,
    max_epochs=50,
    lr=0.1,
    iterator_train__shuffle=True,
    verbose=0
)

param_grid = {
    'lr': [0.01, 0.05, 0.1],
    'module__hidden_units': [10, 20, 30],
    'module__activation_fn': [nn.ReLU(), nn.Tanh(), nn.ELU()] 
}


grid_search = GridSearchCV(
    estimator=net, 
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

print("Starte PyTorch Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# --- Auswertung ---
print("\n--- ERGEBNISSE ---")
print(f"Beste Parameter-Kombination: {grid_search.best_params_}")
print(f"Beste Genauigkeit (Cross-Validation): {grid_search.best_score_:.4f}")