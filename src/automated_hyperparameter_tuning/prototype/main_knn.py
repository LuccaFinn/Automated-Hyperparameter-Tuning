import os
import sys
import pygad
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from SKLearnEncoderDecoder import SKLearnEncoderDecoder
from SKLearnGATuner import SKLearnGATuner
from SKLearnTrainer import SKLearnTrainer

ALGORITHM = "knn"


def main():
    print("Tuning für K-nearest Neighbor gestartet")

    # config (toml) wird geladen
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "resources", "configs", "config_knn.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config

    assert config["Model"]["algorithm"] == ALGORITHM

    # daten laden
    X_train, X_val, y_train, y_val = config_loader.load_data()
    data = (X_train, X_val, y_train, y_val)

    print(f"Daten geladen insgesamt: X_train={X_train.shape}, y_train={y_train.shape}")

    search_space = config["KNN"]["SearchSpace"]
    encoder      = SKLearnEncoderDecoder(ALGORITHM, search_space)
    trainer      = SKLearnTrainer()

    # encode für lesbar
    initial_params  = config["KNN"]["InitialParameters"]
    initial_encoded = encoder.encode(initial_params)

    # der parameter tuner
    ga_tuner = SKLearnGATuner(ALGORITHM, encoder, trainer, data)

    #wo soll er suchen wo kann er rumrödeln?
    gene_space = [
        {"low": search_space["n_neighbors"][0], "high": search_space["n_neighbors"][1]},
        {"low": 0,                               "high": 1},   #0=uniform,1=distance
        {"low": 0,                               "high": 2},   #0=euclidean,1=manhattan,2=minkowski
    ]

    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=4,
        sol_per_pop=8,
        num_genes=3,
        fitness_func=ga_tuner.fitness_func,
        initial_population=[list(initial_encoded) for _ in range(8)],
        gene_space=gene_space,
        mutation_percent_genes=20
    )

    starttime = datetime.now()
    print("GA gestartet um: " + starttime.strftime('%H:%M:%S'))

    ga.run()

    finishtime = datetime.now()
    print("GA beendet um: " + finishtime.strftime('%H:%M:%S'))
    dauer = finishtime - starttime
    print("Dauer insgesamt: " + str(dauer))

    #beste lösung
    solution, fitness, _ = ga.best_solution()
    best_params = encoder.decode(solution)

    print("\n------------------")
    print("Bestes Ergebnis")
    print("Fitness (Accuracy):", fitness)
    print("Parameter", best_params)
    print("------------------\n")

    #speischern also in toml
    config_loader.config["KNN"]["TunedParameters"] = best_params
    config_loader.save(config_path)

    print("Zurück in TOML geschrieben")


if __name__ == "__main__":
    main()
