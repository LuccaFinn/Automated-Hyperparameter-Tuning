import torch
import torch.nn as nn


class Trainer:
    def train(self, model, data, params):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"]
        )

        loss_fn = nn.MSELoss()

        X_train, X_val, y_train, y_val = data

        for _ in range(params["epochs"]):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()

        # BUGFIX: Validierungs-Loss fuer Hyperparameter-Tuning verwenden,
        # nicht den Training-Loss — sonst wird Overfitting belohnt
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = loss_fn(val_output, y_val)

        return -val_loss.item()