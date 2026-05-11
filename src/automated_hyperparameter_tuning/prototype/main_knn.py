import os
import sys
import pygad

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from SKLearnEncoderDecoder import SKLearnEncoderDecoder
from SKLearnGATuner import SKLearnGATuner
from SKLearnTrainer import SKLearnTrainer

ALGORITHM = "knn"


def main():
    print("START MAIN - KNN")

    # 1. CONFIG LADEN
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "resources", "configs", "config_knn.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config

    assert config["Model"]["algorithm"] == ALGORITHM, \
        f"Falsche Config: erwartet '{ALGORITHM}', gefunden '{config['Model']['algorithm']}'"

    print("CONFIG GELADEN")

    # 2. DATEN LADEN
    X_train, X_val, y_train, y_val = config_loader.load_data()
    data = (X_train, X_val, y_train, y_val)

    print(f"DATA LOADED: X_train={X_train.shape}, y_train={y_train.shape}")

    # 3. ENCODER + TRAINER
    search_space = config["KNN"]["SearchSpace"]
    encoder      = SKLearnEncoderDecoder(ALGORITHM, search_space)
    trainer      = SKLearnTrainer()

    # 4. INITIAL PARAMS ENKODIEREN
    initial_params  = config["KNN"]["InitialParameters"]
    initial_encoded = encoder.encode(initial_params)

    # 5. GA TUNER
    ga_tuner = SKLearnGATuner(ALGORITHM, encoder, trainer, data)

    # 6. GENE SPACE
    gene_space = [
        {"low": search_space["n_neighbors"][0], "high": search_space["n_neighbors"][1]},
        {"low": 0,                               "high": 1},   # weights: 0=uniform,1=distance
        {"low": 0,                               "high": 2},   # metric:  0=euclidean,1=manhattan,2=minkowski
    ]

    # 7. GA SETUP + RUN
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

    print("GA STARTED")
    ga.run()
    print("GA FINISHED")

    # 8. BESTE LOESUNG
    solution, fitness, _ = ga.best_solution()
    best_params = encoder.decode(solution)

    print("\n========================")
    print("BEST RESULT - KNN")
    print("Fitness (Accuracy):", fitness)
    print("Params:", best_params)
    print("========================\n")

    # 9. SPEICHERN
    config_loader.config["KNN"]["TunedParameters"] = best_params
    config_loader.save(config_path)

    print("SAVED TO TOML")


if __name__ == "__main__":
    main()
