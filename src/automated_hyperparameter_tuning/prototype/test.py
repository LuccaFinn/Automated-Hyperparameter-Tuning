import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# DATEN LADEN
# ─────────────────────────────────────────────

CSV_PATH = "C:/Users/bjoer/source/repos/Automated-Hyperparameter-Tuning/resources/data/exampleCSVBanana.csv"

df = pd.read_csv(CSV_PATH)

X = df.iloc[:, 0:7].values
y_raw = df.iloc[:, 7]

le = LabelEncoder()
y = le.fit_transform(y_raw).astype(float)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)

# ─────────────────────────────────────────────
# MODELL
# ─────────────────────────────────────────────

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 88),
            nn.Tanh(),
            nn.Linear(88, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

model     = NeuralNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.MSELoss()

print(model)
print()
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Val Acc':>10}")
print("-" * 48)

for epoch in range(161):
    model.train()
    optimizer.zero_grad()
    output = loss_fn(model(X_train), y_train)
    output.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss   = loss_fn(val_output, y_val)
            val_acc    = ((val_output >= 0.5).float() == y_val).float().mean().item()

        print(f"{epoch + 1:>6}  {output.item():>12.4f}  {val_loss.item():>12.4f}  {val_acc * 100:>9.2f}%")

# ─────────────────────────────────────────────
# FINALES ERGEBNIS
# ─────────────────────────────────────────────

model.eval()
with torch.no_grad():
    final_output = model(X_val)
    final_preds  = (final_output >= 0.5).float()
    final_acc    = (final_preds == y_val).float().mean().item()

print()
print("=" * 48)
print("FINALES ERGEBNIS")
print(f"Validation Accuracy: {final_acc * 100:.2f}%")
print(f"Validation Loss:     {loss_fn(final_output, y_val).item():.4f}")
print("=" * 48)