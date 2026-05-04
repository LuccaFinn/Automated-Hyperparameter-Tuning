import toml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ConfigLoader:
    def __init__(self, path):
        self.config_path = Path(path).resolve()

        if not self.config_path.exists():
            raise FileNotFoundError(f"CONFIG NICHT GEFUNDEN: {self.config_path}")

        self.config = toml.load(self.config_path)

        self.absolute_csv_path = Path("C:/Users/bjoer/source/repos/Automated-Hyperparameter-Tuning/resources/data/exampleCSVBanana.csv")

    def get_nn_config(self):
        return self.config.get("NeuralNetwork", {})

    def get_data_config(self):
        return self.config.get("Data", {})

    def load_data(self):
        csv_path = self.absolute_csv_path

        print("CSV Pfad:", csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV ncith gefunden: {csv_path}")

        df = pd.read_csv(csv_path)

        data_config = self.get_data_config()

        start = data_config["input_start"]
        end = data_config["input_end"] + 1
        target_col = data_config["target_column"]

        X = df.iloc[:, start:end].values

        y_raw = df.iloc[:, target_col]
        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(float)


        if np.isnan(y).any():
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=data_config.get("test_size", 0.2),
            random_state=data_config.get("random_state", 42)
        )

        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
        )

    def save(self, path=None):
        target_path = Path(path).resolve() if path else self.config_path

        with open(target_path, "w") as f:
            toml.dump(self.config, f)

        print(f"Config jespeischert unter: {target_path}")