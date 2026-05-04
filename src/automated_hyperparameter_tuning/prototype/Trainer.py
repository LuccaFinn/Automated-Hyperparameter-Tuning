import torch
import torch.nn as nn


class Trainer:
    def get_loss_function(self,name):
        if name == "mse":
            return nn.MSELoss()
        if name == "bce":
            return nn.BCELoss()
        else:
            raise ValueError("Kenn ick net, diese {name}")

    def train(self, model, data, params):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"]
        )

        loss_fn = self.get_loss_function(params["loss_function"])

        X_train, X_val, y_train, y_val = data

        for _ in range(params["epochs"]):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = loss_fn(val_output, y_val)

        #Warum hier n Minus ist ditte kann Herr Wicke gut erklären
        return -val_loss.item()