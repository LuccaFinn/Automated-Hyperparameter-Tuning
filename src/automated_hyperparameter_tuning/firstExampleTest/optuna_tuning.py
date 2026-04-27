import optuna
from training import train


#Optuna optimiert halt wa
def optimize(layers, epochs, X, y, activation_name):

    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 5, 50)
        lr = trial.suggest_float("learn_rate", 1e-4, 1e-1, log=True)
        new_layers = [
            layers[0],
            hidden_size,
            layers[-1]
        ]
        return train(new_layers, lr, 200, activation_name, X, y)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    return study.best_params