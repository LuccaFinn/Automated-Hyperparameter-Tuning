import os
import sys
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import ParameterSampler
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
    print("Random search für SVM")
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
    n_iter      = config["SVM"].get("random_search_iterations", 20)
    best_score  = float("-inf")
    best_params = None
    sampler     = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    print(f"Task:        {task}")
    print(f"Iterationen: {n_iter}")

    for i, params in enumerate(sampler, 1):
        score = train_and_evaluate(params, data, task)
        print(f"[{i}/{n_iter}] {label}: {score:.4f} | Params: {params}")

        if score > best_score:
            best_score  = score
            best_params = params

    print("\n------------------")
    print("Bestes Ergebnis")
    print(f"{label}:    {best_score:.4f}")
    print(f"Parameter: {best_params}")
    print("------------------\n")

    config_loader.config["SVM"]["RandomSearchParameters"] = best_params
    config_loader.save(config_path)
    print("Zürck in TOML geschrieben")


if __name__ == "__main__":
    main()
