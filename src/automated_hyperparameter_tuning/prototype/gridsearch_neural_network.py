import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid

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


def build_fine_grid(best_params, search_space):
    # layer1:
    l1_start, l1_end = search_space["layer1"][0], search_space["layer1"][1]
    coarse_l1_step = max(1, (l1_end - l1_start) // 8)
    l1_range = [max(l1_start, best_params["layer1"] - coarse_l1_step),
                min(l1_end, best_params["layer1"] + coarse_l1_step)]
    fine_l1 = np.linspace(l1_range[0], l1_range[1], 8).astype(int).tolist()
    fine_l1 = sorted(list(set(fine_l1)))

    # layer2:
    l2_start, l2_end = search_space["layer2"][0], search_space["layer2"][1]
    coarse_l2_step = max(1, (l2_end - l2_start) // 8)
    l2_range = [max(l2_start, best_params["layer2"] - coarse_l2_step),
                min(l2_end, best_params["layer2"] + coarse_l2_step)]
    fine_l2 = np.linspace(l2_range[0], l2_range[1], 8).astype(int).tolist()
    fine_l2 = sorted(list(set(fine_l2)))

    # layer3:
    l3_start, l3_end = search_space["layer3"][0], search_space["layer3"][1]
    coarse_l3_step = max(1, (l3_end - l3_start) // 8)
    l3_range = [max(l3_start, best_params["layer3"] - coarse_l3_step),
                min(l3_end, best_params["layer3"] + coarse_l3_step)]
    fine_l3 = np.linspace(l3_range[0], l3_range[1], 8).astype(int).tolist()
    fine_l3 = sorted(list(set(fine_l3)))

    # learning_rate:
    lr_start = max(1e-6, search_space["learning_rate"][0])
    lr_end = max(1e-6, search_space["learning_rate"][1])
    coarse_lr_log_step = (np.log10(lr_end) - np.log10(lr_start)) / 7
    best_lr_log = np.log10(max(1e-6, best_params["learning_rate"]))
    lr_log_range = [max(np.log10(lr_start), best_lr_log - coarse_lr_log_step),
                    min(np.log10(lr_end), best_lr_log + coarse_lr_log_step)]
    fine_lr = np.logspace(lr_log_range[0], lr_log_range[1], 8).tolist()

    # epochs:
    epochs_start, epochs_end = search_space["epochs"][0], search_space["epochs"][1]
    coarse_epochs_step = max(1, (epochs_end - epochs_start) // 8)
    epochs_range = [max(epochs_start, best_params["epochs"] - coarse_epochs_step),
                    min(epochs_end, best_params["epochs"] + coarse_epochs_step)]
    fine_epochs = np.linspace(epochs_range[0], epochs_range[1], 8).astype(int).tolist()
    fine_epochs = sorted(list(set(fine_epochs)))

    return {
        "layer1":        fine_l1,
        "layer2":        fine_l2,
        "layer3":        fine_l3,
        "activation":    [best_params["activation"]],
        "learning_rate": fine_lr,
        "epochs":        fine_epochs,
        "loss_function": [best_params["loss_function"]]
    }




def main():
    print("\n------------------")
    print("Grid search für neuronale Netze UND LOS!")
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

    # === PHASE 1: COARSE GRID SEARCH ===
    print("\n>>> STARTE PHASE 1: GROBE GRID-SUCHE (Coarse Grid Search) <<<")
    param_grid_coarse = build_param_grid(search_space)
    best_loss_coarse  = float("inf")
    best_params_coarse = None
    total_coarse      = len(list(ParameterGrid(param_grid_coarse)))

    for i, params in enumerate(ParameterGrid(param_grid_coarse), 1):
        loss = train_and_evaluate(params, data, input_size, task)
        print(f"[Grob: {i}/{total_coarse}] Loss: {loss:.4f} | Params: {params}")

        if loss < best_loss_coarse:
            best_loss_coarse   = loss
            best_params_coarse = params

    print("\n--- Ergebnis der Grob-Suche ---")
    print(f"Bester Grob-Loss:   {best_loss_coarse:.4f}")
    print(f"Beste Grob-Parameter: {best_params_coarse}")
    print("--------------------------------\n")

    if best_params_coarse is None or best_loss_coarse == float("inf"):
        print("Fehler: Keine gültigen Parameter in der Grob-Suche gefunden!")
        return

    global_best_loss = best_loss_coarse
    global_best_params = best_params_coarse
    source_phase = "Grobe Grid-Suche"

    # === PHASE 2: FINE GRID SEARCH ===
    print(">>> STARTE PHASE 2: FEINE GRID-SUCHE (Fine Grid Search) <<<")
    param_grid_fine = build_fine_grid(best_params_coarse, search_space)
    best_loss_fine  = float("inf")
    best_params_fine = None
    total_fine      = len(list(ParameterGrid(param_grid_fine)))

    for i, params in enumerate(ParameterGrid(param_grid_fine), 1):
        loss = train_and_evaluate(params, data, input_size, task)
        print(f"[Fein: {i}/{total_fine}] Loss: {loss:.4f} | Params: {params}")

        if loss < best_loss_fine:
            best_loss_fine   = loss
            best_params_fine = params

    print("\n--- Ergebnis der Fein-Suche ---")
    print(f"Bester Fein-Loss:   {best_loss_fine:.4f}")
    print(f"Beste Fein-Parameter: {best_params_fine}")
    print("--------------------------------\n")

    if best_loss_fine < global_best_loss:
        global_best_loss = best_loss_fine
        global_best_params = best_params_fine
        source_phase = "Feine Grid-Suche"

    print("\n------------------")
    print("GLOBAL BESTES ERGEBNIS")
    print(f"Bester Loss:   {global_best_loss:.4f}")
    print(f"Parameter:     {global_best_params}")
    print(f"Gefunden in:   {source_phase}")
    print("------------------\n")

    config_loader.config["NeuralNetwork"]["GridSearchParameters"] = global_best_params
    config_loader.save(config_path)

    print(f"Zurück in TOML geschrieben (Ergebnis aus: {source_phase})")



if __name__ == "__main__":
    main()
