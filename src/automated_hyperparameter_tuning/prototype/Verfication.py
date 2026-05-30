import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader


#------------------------------------
#neuronale netze
#------------------------------------

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
        else:
            raise ValueError(f"Unbekannte Aktivierungsfunktion: {name}")

    def forward(self, x):
        return self.model(x)


def verify_neural_network(config_path):
    print("\n------------------------------------")
    print("Verifikation Neural Network")
    print("------------------------------------")

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config
    task          = config["Model"].get("task", "classification")
    params        = config["NeuralNetwork"]["TunedParameters"]

    X_train, X_val, y_train, y_val = config_loader.load_data()

    input_size = X_train.shape[1]
    model      = NeuralNet(params, input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn   = nn.MSELoss() if params["loss_function"] == "mse" else nn.BCEWithLogitsLoss()

    for _ in range(params["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = loss_fn(model(X_train), y_train)
        out.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss   = loss_fn(val_output, y_val).item()

    print(f"Tuned Params:    {params}")
    print(f"Task:            {task}")
    print(f"Validation Loss: {val_loss:.4f}")

    if task == "classification":
        preds = (val_output >= 0.5).float()
        acc   = (preds == y_val).float().mean().item()
        print(f"Validation Accuracy: {acc * 100:.2f}%")
    else:
        print(f"Validation MSE: {val_loss:.4f}")


#------------------------------------
#helper func
#------------------------------------

def to_numpy(data):
    X_train, X_val, y_train, y_val = data
    if hasattr(X_train, "numpy"):
        X_train = X_train.numpy()
        X_val   = X_val.numpy()
        y_train = y_train.numpy().ravel()
        y_val   = y_val.numpy().ravel()
    return X_train, X_val, y_train, y_val


def print_sklearn_results(task, params, y_val, predictions):
    print(f"Tuned Params: {params}")
    print(f"Task:         {task}")

    if task == "regression":
        mse = mean_squared_error(y_val, predictions)
        print(f"Validation MSE: {mse:.4f}")
    else:
        f1  = f1_score(y_val, predictions, average="weighted")
        acc = accuracy_score(y_val, predictions)
        print(f"Validation F1-Score: {f1:.4f}")
        print(f"Validation Accuracy: {acc * 100:.2f}%")


#------------------------------------
#svm
#------------------------------------

def verify_svm(config_path):
    print("\n------------------------------------")
    print("Verifikation SVM")
    print("------------------------------------")

    config_loader                  = ConfigLoader(config_path)
    task                           = config_loader.config["Model"].get("task", "classification")
    params                         = config_loader.config["SVM"]["TunedParameters"]
    X_train, X_val, y_train, y_val = to_numpy(config_loader.load_data())

    if task == "regression":
        model = SVR(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])
    else:
        model = SVC(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])

    model.fit(X_train, y_train)
    print_sklearn_results(task, params, y_val, model.predict(X_val))


#------------------------------------
#knn (k nearest neigbors halt)
#------------------------------------

def verify_knn(config_path):
    print("\n------------------------------------")
    print("Verifikation KNN")
    print("------------------------------------")

    config_loader                  = ConfigLoader(config_path)
    task                           = config_loader.config["Model"].get("task", "classification")
    params                         = config_loader.config["KNN"]["TunedParameters"]
    X_train, X_val, y_train, y_val = to_numpy(config_loader.load_data())

    if task == "regression":
        model = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"]
        )
    else:
        model = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"]
        )

    model.fit(X_train, y_train)
    print_sklearn_results(task, params, y_val, model.predict(X_val))


#------------------------------------
#logreg
#------------------------------------

def verify_logistic_regression(config_path):
    print("\n------------------------------------")
    print("Verifikation LogReg")
    print("------------------------------------")

    config_loader                  = ConfigLoader(config_path)
    task                           = config_loader.config["Model"].get("task", "classification")
    params                         = config_loader.config["LogisticRegression"]["TunedParameters"]
    X_train, X_val, y_train, y_val = to_numpy(config_loader.load_data())

    # Logistische Regression ist immer Klassifikation
    model = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        solver=params["solver"]
    )
    model.fit(X_train, y_train)
    print_sklearn_results("classification", params, y_val, model.predict(X_val))

#------------------------------------
#xgb
#------------------------------------

def verify_xgboost(config_path):
    print("\n ------------------------------------")
    print("Verifikation XGB")
    print("------------------------------------")

    config_loader                  = ConfigLoader(config_path)
    task                           = config_loader.config["Model"].get("task", "classification")
    params                         = config_loader.config["XGBoost"]["TunedParameters"]
    X_train, X_val, y_train, y_val = to_numpy(config_loader.load_data())

    if task == "regression":
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            verbosity=0
        )
    else:
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            eval_metric="logloss",
            verbosity=0
        )

    model.fit(X_train, y_train)
    print_sklearn_results(task, params, y_val, model.predict(X_val))


def build_path(filename):
    return os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "resources", "configs", filename
    )


if __name__ == "__main__":
    verify_neural_network(build_path("config_neural_network.toml"))
    verify_svm(build_path("config_svm.toml"))
    verify_knn(build_path("config_knn.toml"))
    verify_logistic_regression(build_path("config_logistic_regression.toml"))
    verify_linear_regression(build_path("config_linear_regression.toml"))
    verify_xgboost(build_path("config_xgboost.toml"))