import os
import sys
from datetime import datetime

import pygad

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from SKLearnEncoderDecoder import SKLearnEncoderDecoder
from SKLearnGATuner import SKLearnGATuner
from SKLearnTrainer import SKLearnTrainer

ALGORITHM = "xgboost"


def main():
    print("Tuning für XGB gestartet")

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "resources", "configs", "config_xgboost.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config

    assert config["Model"]["algorithm"] == ALGORITHM, \
        f"Falsche Config: erwartet '{ALGORITHM}', gefunden '{config['Model']['algorithm']}'"

    X_train, X_val, y_train, y_val = config_loader.load_data()
    data = (X_train, X_val, y_train, y_val)

    print(f"Daten geladen insgesamt: X_train={X_train.shape}, y_train={y_train.shape}")

    search_space = config["XGBoost"]["SearchSpace"]
    encoder      = SKLearnEncoderDecoder(ALGORITHM, search_space)
    task = config["Model"].get("task", "classification")
    trainer = SKLearnTrainer(task=task)

    initial_params  = config["XGBoost"]["InitialParameters"]
    initial_encoded = encoder.encode(initial_params)

    ga_tuner = SKLearnGATuner(ALGORITHM, encoder, trainer, data)

    gene_space = [
        {"low": search_space["n_estimators"][0],  "high": search_space["n_estimators"][1]},
        {"low": search_space["max_depth"][0],      "high": search_space["max_depth"][1]},
        {"low": search_space["learning_rate"][0],  "high": search_space["learning_rate"][1]},
        {"low": search_space["subsample"][0],      "high": search_space["subsample"][1]},
    ]

    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=4,
        sol_per_pop=8,
        num_genes=4,
        fitness_func=ga_tuner.fitness_func,
        initial_population=[list(initial_encoded) for _ in range(8)],
        gene_space=gene_space,
        mutation_percent_genes=50
    )

    starttime = datetime.now()
    print("GA gestartet um: " + starttime.strftime('%H:%M:%S'))

    ga.run()

    finishtime = datetime.now()
    print("GA beendet um: " + finishtime.strftime('%H:%M:%S'))
    dauer = finishtime - starttime
    print("Dauer insgesamt: " + str(dauer))

    solution, fitness, _ = ga.best_solution()
    best_params = encoder.decode(solution)

    print("\n------------------")
    print("Bestes Ergebnis")
    print("Fitness (F1-Score):", fitness)
    print("Parameter:", best_params)
    print("------------------\n")

    config_loader.config["XGBoost"]["TunedParameters"] = best_params
    config_loader.save(config_path)

    print("Zurück in TOML geschrieben")


if __name__ == "__main__":
    main()
