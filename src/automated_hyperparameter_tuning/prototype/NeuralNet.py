import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, params, input_size):
        super().__init__()

        layers = []
        sizes = [
            input_size,
            params["layer1"],
            params["layer2"],
            params["layer3"],
            1  # BUGFIX: Output-Layer fehlte komplett
        ]

        activation = self.get_activation(params["activation"])

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # Keine Aktivierung nach dem letzten Layer (Output)
            if i < len(sizes) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)

    def get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unbekannte Aktivierungsfunktion: {name}")

    def forward(self, x):
        return self.model(x)