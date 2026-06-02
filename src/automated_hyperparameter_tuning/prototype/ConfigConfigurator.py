import os
import sys
from datetime import datetime
import time

from pandas._libs import algos

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader


def main():
    print("Config Configuration gestartet")

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "resources", "configs", "config.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config

    algorithm = config["Algorithm"]

    # Festlegen der initialen parameter hier, so müssen die nicht von Gruppe 1 angegeben werden - UNFINISHED!
    if algorithm == "knn":

        grenzeLayer1 = abs(config["Data"]["input_start"] - config["Data"]["input_end"] - 1)
        grenzeLayer2 = config["Data"]["input_end"] - config["Data"]["target_column"]

        search_space = {
            "layer1": [1, grenzeLayer1],
            "layer2": [1, 500],
            "layer3": [2, grenzeLayer2],
            "activation": [ "relu", "tanh", "sigmoid",],
            "learning_rate": [ 1e-5, 1.0,],
            "epochs": [1, 500,],
            "loss_function": [ "mse", "bce", "mae"]
        }

        config_loader.config["NeuralNetwork"]["SearchSpace"] = search_space
        config_loader.save(config_path)

    if algorithm == "logistic_regression":
        search_space = {
            "C": [0.001, 100.0],
            "max_iter": [100, 1000],
            "solver": ["lbfgs", "saga"],
            "penalty": ["l2", "none"]
        }
        config_loader.config["LogisticRegression"]["SearchSpace"] = search_space
        config_loader.save(config_path)

    if algorithm == "svm":
        search_space = {
            "C" : [0.01, 100.0, ],
            "kernel" : ["linear", "rbf", "poly", "sigmoid", ],
            "gamma" : ["scale", "auto", ]
        }
        config_loader.config["SVM"]["SearchSpace"] = search_space
        config_loader.save(config_path)

    if algorithm == "knn":
        search_space = {
            "n_neighbors" : [1, 50, ],
            "weights" : ["uniform", "distance", ],
            "metric" : ["euclidean", "manhattan", "minkowski", ]
        }
        config_loader.config["KNN"]["SearchSpace"] = search_space
        config_loader.save(config_path)