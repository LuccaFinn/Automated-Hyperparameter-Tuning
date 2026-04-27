#Das einzige woran der Kerl derzeit juckelt sind die HiddenSize und die Learnrate

import torch
import toml

from config_loader import load_config
from optuna_tuning import optimize

#Der lädt die Config mit der Configloader.py - Also die Methode benutzt er halt
config, config_path = load_config()

initial = config["InitialParameters"]

layers = initial["net_layers"]
lr = initial["learn_rate"]
epochs = initial["epochs"]
activation = initial["activation_function"]

#Dummy Daten, also später brauchen wir hier halt noch nen CSV Reader, damit wir hier halt die echten Daten verarbeiten kÖnnen
X = torch.randn(100, layers[0])
y = torch.randn(100, layers[-1])


#Im Optimalfall optimiert er hier also halt mit Optuna. Das Müssen wir dann halt durch unsere Lösung ersetzen
best = optimize(layers, epochs, X, y, activation)

print("Beste Parameter: ", best)

# Das sollte zurückschreiben, der Bumms will aber noch nicht so wirklich - Also doch aber löscht halt die Erklärungen(Kommentare) in der TOML
#Deswegen sind die Kommentare auch in der Toml weg
config["TunedParameters"]["net_layers"] = [
    layers[0],
    best["hidden_size"],
    layers[-1]
]

#Hier schreibt er dann den Rest, also mit dem dump

config["TunedParameters"]["learn_rate"] = best["learn_rate"]

with open(config_path, "w") as f:
    toml.dump(config, f)

print("Fertig")