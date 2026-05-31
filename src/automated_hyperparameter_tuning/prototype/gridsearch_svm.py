import os
import sys
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader


def to_numpy(data):
    X_train, X_val, y_train, y_val = data
    if hasattr(X_train, "numpy"):
        X_train = X_train.numpy()
        X_val   = X_val.numpy()
        y_train = y_train.numpy().ravel()
        y_val   = y_val.numpy().ravel()
    return X_train, X_val, y_train, y_val


def train_and_evaluate(params, data, task):
    X_train, X_val, y_train, y_val = data

    if task == "regression":
        model = SVR(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])
    else:
        model = SVC(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    if task == "regression":
        return -mean_squared_error(y_val, predictions)
    else:
        return f1_score(y_val, predictions, average="weighted")


def build_param_grid(search_space):
    return {
        "C":      np.linspace(search_space["C"][0], search_space["C"][1], 5).tolist(),
        "kernel": search_space["kernel"],
        "gamma":  search_space["gamma"]
    }


def main():
    print("\n------------------")
    print("Grid search für SVM")
    print("------------------")

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "resources", "configs", "config_svm.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config
    task          = config["Model"].get("task", "classification")
    search_space  = config["SVM"]["SearchSpace"]

    data  = to_numpy(config_loader.load_data())
    label = "MSE" if task == "regression" else "F1"

    param_grid  = build_param_grid(search_space)
    best_score  = float("-inf")
    best_params = None
    total       = len(list(ParameterGrid(param_grid)))

    #print(f"Task:          {task}")
    #print(f"Kombinationen: {total}")

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        score = train_and_evaluate(params, data, task)
        #print(f"[{i}/{total}] {label}: {score:.4f} | Params: {params}")

        if score > best_score:
            best_score  = score
            best_params = params

    print("\n------------------")
    print("Bestes Ergebnis")
    print(f"{label}:    {best_score:.4f}")
    print(f"Parameter: {best_params}")
    print("------------------\n")

    config_loader.config["SVM"]["GridSearchParameters"] = best_params
    config_loader.save(config_path)
    print("Zurück ion TOML geschrieben")


if __name__ == "__main__":
    main()
