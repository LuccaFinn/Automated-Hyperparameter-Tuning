import torch
import torch.nn as nn


class Trainer:
    def __init__(self, task="classification", early_stopping=True, patience=10, min_delta=1e-4):
        self.task           = task
        self.early_stopping = early_stopping
        self.patience       = patience
        self.min_delta      = min_delta

    def get_loss_fn(self, name):
        if name == "mse":
            return nn.MSELoss()
        elif name == "mae":
            return nn.L1Loss()
        elif name == "bce":
            return nn.BCEWithLogitsLoss()

    def train(self, model, data, params):
        if self.task == "regression" and params["loss_function"] == "bce":
            return -float("inf")

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn   = self.get_loss_fn(params["loss_function"])


        X_train, X_val, y_train, y_val = data

        best_val_loss     = float("inf")
        epochs_no_improve = 0
        best_weights      = None

        for epoch in range(params["epochs"]):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                best_weights      = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if self.early_stopping and epochs_no_improve >= self.patience:
                if best_weights is not None:
                    model.load_state_dict(best_weights)
                break


        if best_weights is not None:
            model.load_state_dict(best_weights)

        return -best_val_loss