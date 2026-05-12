import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error


class SKLearnTrainer:
    def get_model(self, algorithm, params):
        if algorithm == "svm":
            return SVC(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params["gamma"]
            )
        elif algorithm == "knn":
            return KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                metric=params["metric"]
            )
        elif algorithm == "logistic_regression":
            return LogisticRegression(
                C=params["C"],
                max_iter=params["max_iter"],
                solver=params["solver"],
                penalty=params["penalty"]
            )
        elif algorithm == "linear_regression":
            return LinearRegression()
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

        if algorithm == "linear_regression":
            #Wieder gleicher Bumms wegen MSE negative Fitness, wie beim nn halt
            mse = mean_squared_error(y_val, predictions)
            return -mse, model
        else:
            #bei klassifiatkino accuracy als fitness
            acc = accuracy_score(y_val, predictions)
            return acc, model
