import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor


class SKLearnTrainer:
    def __init__(self, task="classification"):
        self.task = task

    def get_model(self, algorithm, params):
        if algorithm == "svm":
            if self.task == "regression":
                return SVR(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])
            else:
                return SVC(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])
        elif algorithm == "knn":
            if self.task == "regression":
                return KNeighborsRegressor(
                    n_neighbors=params["n_neighbors"],
                    weights=params["weights"],
                    metric=params["metric"]
                )
            else:
                return KNeighborsClassifier(
                    n_neighbors=params["n_neighbors"],
                    weights=params["weights"],
                    metric=params["metric"]
                )
        elif algorithm == "logistic_regression":
            # Logreg immer classifikation
            return LogisticRegression(
                C=params["C"],
                max_iter=params["max_iter"],
                solver=params["solver"]
            )
        elif algorithm == "xgboost":
            if self.task == "regression":
                return XGBRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params["subsample"],
                    verbosity=0
                )
            else:
                return XGBClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params["subsample"],
                    eval_metric="logloss",
                    verbosity=0
                )
        else:
            raise ValueError(f"Unbekannter Algorithmus: {algorithm}")

    def train(self, algorithm, data, params):
        X_train, X_val, y_train, y_val = data

        if hasattr(X_train, "numpy"):
            X_train = X_train.numpy()
            X_val   = X_val.numpy()
            y_train = y_train.numpy().ravel()
            y_val   = y_val.numpy().ravel()

        model = self.get_model(algorithm, params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        if self.task == "regression":
            mse = mean_squared_error(y_val, predictions)
            return -mse, model
        else:
            score = f1_score(y_val, predictions, average="weighted")
            return score, model