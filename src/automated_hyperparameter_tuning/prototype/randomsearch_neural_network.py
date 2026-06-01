import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterSampler

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader


class NeuralNet(nn.Module):
    def __init__(self, params, input_size):
        super().__init__()

        sizes = [
            input_size,
            params["layer1"],
            params["layer2"],
            params["layer3"],
            1
        ]

        activation = self.get_activation(params["activation"])

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
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

    def forward(self, x):
        return self.model(x)


def get_loss_fn(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "bce":
        return nn.BCEWithLogitsLoss()


def train_and_evaluate(params, data, input_size, task, early_stopping=True, patience=10, min_delta=1e-4):
    if task == "regression" and params["loss_function"] == "bce":
        return float("inf")

    # Fix seed for reproducibility/consistency across different grid points
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, X_val, y_train, y_val = data

    model     = NeuralNet(params, input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn   = get_loss_fn(params["loss_function"])



    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_weights      = None

    for _ in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val_loss - min_delta:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_weights      = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if early_stopping and epochs_no_improve >= patience:
            if best_weights is not None:
                model.load_state_dict(best_weights)
            break


    if best_weights is not None:
        model.load_state_dict(best_weights)

    return best_val_loss


def build_param_grid(search_space):
    lr_start = max(1e-6, search_space["learning_rate"][0])
    lr_end = max(1e-6, search_space["learning_rate"][1])
    return {
        "layer1":        list(range(search_space["layer1"][0],        search_space["layer1"][1] + 1,        max(1, (search_space["layer1"][1]  - search_space["layer1"][0])  // 8))),
        "layer2":        list(range(search_space["layer2"][0],        search_space["layer2"][1] + 1,        max(1, (search_space["layer2"][1]  - search_space["layer2"][0])  // 8))),
        "layer3":        list(range(search_space["layer3"][0],        search_space["layer3"][1] + 1,        max(1, (search_space["layer3"][1]  - search_space["layer3"][0])  // 8))),
        "activation":    search_space["activation"],
        "learning_rate": np.logspace(np.log10(lr_start), np.log10(lr_end), 8).tolist(),
        "epochs":        list(range(search_space["epochs"][0],        search_space["epochs"][1] + 1,        max(1, (search_space["epochs"][1]  - search_space["epochs"][0])  // 8))),
        "loss_function": search_space["loss_function"]
    }



def main():
    print("\n------------------")
    print("Random search für neuronale Netz")
    print("------------------")

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "resources", "configs", "config_neural_network_algo.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config
    task          = config["Model"].get("task", "classification")
    nn_config     = config_loader.get_nn_config()
    search_space  = nn_config["SearchSpace"]

    X_train, X_val, y_train, y_val = config_loader.load_data()
    data       = (X_train, X_val, y_train, y_val)
    input_size = X_train.shape[1]

    param_grid  = build_param_grid(search_space)
    n_iter      = config["NeuralNetwork"].get("random_search_iterations", 20)
    best_loss   = float("inf")
    best_params = None
    sampler     = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    #print(f"Task:        {task}")
    #print(f"Iterationen: {n_iter}")

    for i, params in enumerate(sampler, 1):
        loss = train_and_evaluate(params, data, input_size, task)

        print(f"[{i}/{n_iter}] Loss: {loss:.4f} | Params: {params}")

        if loss < best_loss:
            best_loss   = loss
            best_params = params

    print("\n------------------")
    print("Bestes Ergebnis")
    print(f"Loss:   {best_loss:.4f}")
    print(f"Parameter: {best_params}")
    print("------------------\n")

    config_loader.config["NeuralNetwork"]["RandomSearchParameters"] = best_params
    config_loader.save(config_path)
    print("Zurück in TOML geschrieben")


if __name__ == "__main__":
    main()
