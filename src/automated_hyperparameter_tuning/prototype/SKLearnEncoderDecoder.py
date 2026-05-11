class SKLearnEncoderDecoder:
    def __init__(self, algorithm, search_space):
        self.algorithm = algorithm

        # ── SVM ──────────────────────────────────────────
        self.svm_kernel_map = {0: "linear", 1: "rbf", 2: "poly", 3: "sigmoid"}
        self.svm_gamma_map  = {0: "scale", 1: "auto"}
        self.svm_kernel_rev = {v: k for k, v in self.svm_kernel_map.items()}
        self.svm_gamma_rev  = {v: k for k, v in self.svm_gamma_map.items()}

        # ── KNN ──────────────────────────────────────────
        self.knn_weights_map = {0: "uniform", 1: "distance"}
        self.knn_metric_map  = {0: "euclidean", 1: "manhattan", 2: "minkowski"}
        self.knn_weights_rev = {v: k for k, v in self.knn_weights_map.items()}
        self.knn_metric_rev  = {v: k for k, v in self.knn_metric_map.items()}

        # ── Logistische Regression ────────────────────────
        self.lr_solver_map  = {0: "lbfgs", 1: "saga"}
        self.lr_penalty_map = {0: "l2", 1: "none"}
        self.lr_solver_rev  = {v: k for k, v in self.lr_solver_map.items()}
        self.lr_penalty_rev = {v: k for k, v in self.lr_penalty_map.items()}

    # ─────────────────────────────────────────────────────
    # ENCODE
    # ─────────────────────────────────────────────────────

    def encode(self, params):
        if self.algorithm == "svm":
            return [
                params["C"],
                self.svm_kernel_rev[params["kernel"]],
                self.svm_gamma_rev[params["gamma"]]
            ]
        elif self.algorithm == "knn":
            return [
                params["n_neighbors"],
                self.knn_weights_rev[params["weights"]],
                self.knn_metric_rev[params["metric"]]
            ]
        elif self.algorithm == "logistic_regression":
            return [
                params["C"],
                params["max_iter"],
                self.lr_solver_rev[params["solver"]],
                self.lr_penalty_rev[params["penalty"]]
            ]
        elif self.algorithm == "linear_regression":
            # Lineare Regression hat keine Hyperparameter zum Tunen
            return [0]
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algorithm}")

    # ─────────────────────────────────────────────────────
    # DECODE
    # ─────────────────────────────────────────────────────

    def decode(self, solution):
        if self.algorithm == "svm":
            return {
                "C":      float(solution[0]),
                "kernel": self.svm_kernel_map[int(solution[1])],
                "gamma":  self.svm_gamma_map[int(solution[2])]
            }
        elif self.algorithm == "knn":
            return {
                "n_neighbors": max(1, int(solution[0])),
                "weights":     self.knn_weights_map[int(solution[1])],
                "metric":      self.knn_metric_map[int(solution[2])]
            }
        elif self.algorithm == "logistic_regression":
            return {
                "C":        float(solution[0]),
                "max_iter": max(100, int(solution[1])),
                "solver":   self.lr_solver_map[int(solution[2])],
                "penalty":  self.lr_penalty_map[int(solution[3])]
            }
        elif self.algorithm == "linear_regression":
            return {}
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algorithm}")
