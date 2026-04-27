import torch.nn as nn
from utils import get_activation

class Net(nn.Module):
    def __init__(self, layers, activation_name):
        super().__init__()

        modules = []
        activation = get_activation(activation_name)

        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))

            if i < len(layers) - 2:
                modules.append(activation)

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)