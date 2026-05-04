import os
import sys
import pygad
import matplotlib.pyplot as plt
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from EncoderDecoder import EncoderDecoder
from GATuner import GATuner
from Trainer import Trainer


def main():

    print("Ab geht die wilde Fahrt, alle mal anschnallen!")
    #config laden
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "resources",
        "configs",
        "configForPrototype.toml"
    )

    config_loader = ConfigLoader(config_path)
    nn_config = config_loader.get_nn_config()

    #print("Config geladen")

    #daten laden
    X_train, X_val, y_train, y_val = config_loader.load_data()

    #print(f"{X_train.shape}, {y_train.shape}")
    if X_train.size(0) == 0 or y_train.size(0) == 0:
        print("Fehler: Deine CSV ist alle!")
        return

    print(f"Daten geladen insgesamt: X_train={X_train.shape}, y_train={y_train.shape}")

    data = (X_train, X_val, y_train, y_val)
    input_size = X_train.shape[1]

    encoder = EncoderDecoder(nn_config["SearchSpace"])
    trainer = Trainer()

    print("Starte sofort")
    time.sleep(3)
    print("Sorry Chef, Sekundenschlaf... Geht los!\n")

    #initialen parameter aus config
    initial_params = nn_config["InitialParameters"]
    initial_encoded = encoder.encode(initial_params)

    #print("Initiale Parameter encoded")

    ga_tuner = GATuner(
        encoder=encoder,
        trainer=trainer,
        data=data,
        input_size=input_size
    )

    #gene_space definiert die erlaubten Wertebereiche pro Gen - ohne erzeugt PYGad zufälliuge Flots
    search_space = nn_config["SearchSpace"]
    gene_space = [
        {"low": search_space["layer1"][0], "high": search_space["layer1"][1]},
        {"low": search_space["layer2"][0], "high": search_space["layer2"][1]},
        {"low": search_space["layer3"][0], "high": search_space["layer3"][1]},
        {"low": 0, "high": 2},
        {"low": search_space["learning_rate"][0], "high": search_space["learning_rate"][1]},
        {"low": search_space["epochs"][0], "high": search_space["epochs"][1]},
        {"low": 0, "high": 1},
    ]

    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=4,
        sol_per_pop=8,
        num_genes=7,
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

    print("\n------------------")
    print("Bestes Ergebnis")
    print("Solution:", solution)
    #print("Fitness:", fitness)
    print("Loss:", -fitness)
    print("------------------\n")

    best_params = encoder.decode(solution)

    #print("Decoded Parameter:", best_params)

    #toml writeback
    config_loader.config["NeuralNetwork"]["TunedParameters"] = best_params
    config_loader.save(config_path)

    print("Zurück in TOML geschrieben")


if __name__ == "__main__":
    main()