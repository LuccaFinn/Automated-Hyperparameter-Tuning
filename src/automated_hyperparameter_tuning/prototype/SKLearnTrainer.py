import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error


class SKLearnTrainer:
    def normalize_algorithm(self, algorithm):
        algo = algorithm.lower().replace("_", "")
        if algo == "knn":
            return "knn"
        elif algo == "svm":
            return "svm"
        elif algo in ["logisticregression", "lr"]:
            return "logistic_regression"
        elif algo == "linearregression":
            return "linear_regression"
        elif algo in ["randomforest", "rf"]:
            return "random_forest"
        elif algo in ["xgboost", "xgb"]:
            return "xgboost"
        elif algo in ["mlpclassifier", "mlp"]:
            return "mlp_classifier"
        else:
            return algorithm.lower()

    def get_model(self, algorithm, params):
        algo = self.normalize_algorithm(algorithm)
        if algo == "svm":
            return SVC(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params["gamma"]
            )
        elif algo == "knn":
            return KNeighborsClassifier(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                metric=params["metric"]
            )
        elif algo == "logistic_regression":
            return LogisticRegression(
                C=params["C"],
                max_iter=params["max_iter"],
                solver=params["solver"],
                penalty=params["penalty"]
            )
        elif algo == "linear_regression":
            return LinearRegression()
        elif algo == "random_forest":
            max_depth = params.get("max_depth")
            if max_depth == "None" or max_depth is None:
                max_depth = None
            else:
                max_depth = int(max_depth)
            return RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=max_depth
            )
        elif algo == "xgboost":
            learning_rate = float(params.get("learning_rate", 0.1))
            max_depth = params.get("max_depth", 3)
            max_depth = None if (max_depth == "None" or max_depth is None) else int(max_depth)
            n_estimators = int(params.get("n_estimators", 100))
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    eval_metric="logloss"
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth if max_depth is not None else 3,
                    n_estimators=n_estimators
                )
        elif algo == "mlp_classifier":
            from sklearn.neural_network import MLPClassifier
            activation = params.get("activation", "relu")
            layer1 = int(params.get("layer1", 100))
            layer2 = params.get("layer2")
            if layer2 is not None and layer2 != "None":
                hidden_layer_sizes = (layer1, int(layer2))
            else:
                hidden_layer_sizes = (layer1,)
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=float(params.get("learning_rate", 0.001)),
                max_iter=int(params.get("epochs", 200))
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

        algo = self.normalize_algorithm(algorithm)
        if algo == "linear_regression":
            #Wieder gleicher Bumms wegen MSE negative Fitness, wie beim nn halt
            mse = mean_squared_error(y_val, predictions)
            return -mse, model
        else:
            #bei klassifiatkino accuracy als fitness
            acc = accuracy_score(y_val, predictions)
            return acc, model
