import os
import sys
import pygad

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from SKLearnEncoderDecoder import SKLearnEncoderDecoder
from SKLearnGATuner import SKLearnGATuner
from SKLearnTrainer import SKLearnTrainer
from datetime import datetime

ALGORITHM = "svm"

#Es ist alles schema F, immer das gleich nur für den jeweiligen algo halt angeopasst
def main():
    print("Tuning für svm gestartet")

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "resources", "configs", "config_svm.toml"
    )

    config_loader = ConfigLoader(config_path)
    config        = config_loader.config

    X_train, X_val, y_train, y_val = config_loader.load_data()
    data = (X_train, X_val, y_train, y_val)

    print(f"Daten geladen insgesamt: X_train={X_train.shape}, y_train={y_train.shape}")

    search_space = config["SVM"]["SearchSpace"]
    encoder      = SKLearnEncoderDecoder(ALGORITHM, search_space)
    trainer      = SKLearnTrainer()

    initial_params  = config["SVM"]["InitialParameters"]
    initial_encoded = encoder.encode(initial_params)

    ga_tuner = SKLearnGATuner(ALGORITHM, encoder, trainer, data)

    gene_space = [
        {"low": search_space["C"][0],      "high": search_space["C"][1]},
        {"low": 0,                          "high": 3},   #kernel:0=linear,1=rbf,2=poly,3=sigmoid
        {"low": 0,                          "high": 1},   #gamma:0=scale,1=auto
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

    solution, fitness, _ = ga.best_solution()
    best_params = encoder.decode(solution)

    print("\n------------------")
    print("Bestes Ergebnis")
    print("Fitness (Accuracy):", fitness)
    print("Paramameter:", best_params)
    print("------------------\n")

    config_loader.config["SVM"]["TunedParameters"] = best_params
    config_loader.save(config_path)

    print("Zurück in TOML geschrieben")


if __name__ == "__main__":
    main()
