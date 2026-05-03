import os
import sys
import pygad
import matplotlib.pyplot as plt
from datetime import datetime

# BUGFIX: sys.path anpassen, damit lokale Module gefunden werden
sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from EncoderDecoder import EncoderDecoder
from GATuner import GATuner
from Trainer import Trainer


def main():

    print("START MAIN")

    # 1. CONFIG LADEN
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

    print("CONFIG GELADEN")

    # 2. DATEN LADEN
    X_train, X_val, y_train, y_val = config_loader.load_data()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_train.size(0) == 0 or y_train.size(0) == 0:
        print("ERROR: Empty training data!")
        return

    print(f"DATA LOADED: X_train={X_train.shape}, y_train={y_train.shape}")

    data = (X_train, X_val, y_train, y_val)
    input_size = X_train.shape[1]

    # 3. ENCODER + TRAINER
    encoder = EncoderDecoder(nn_config["SearchSpace"])
    trainer = Trainer()

    print("MODELS READY")

    # 4. INITIAL PARAMS
    initial_params = nn_config["InitialParameters"]
    initial_encoded = encoder.encode(initial_params)

    print("INITIAL PARAMS ENCODED")

    # 5. GA TUNER
    ga_tuner = GATuner(
        encoder=encoder,
        trainer=trainer,
        data=data,
        input_size=input_size
    )

    # BUGFIX: gene_space definiert die erlaubten Wertebereiche pro Gen.
    # Ohne gene_space wuerde PyGAD zufaellige Floats in [0,1] erzeugen,
    # was fuer Integer-Parameter (layer-Groessen, Epochen) und den
    # Aktivierungs-Index (0-2) komplett falsch waere.
    search_space = nn_config["SearchSpace"]
    gene_space = [
        {"low": search_space["layer1"][0],       "high": search_space["layer1"][1]},
        {"low": search_space["layer2"][0],       "high": search_space["layer2"][1]},
        {"low": search_space["layer3"][0],       "high": search_space["layer3"][1]},
        {"low": 0,                               "high": 2},   # Aktivierungs-Index (0=relu,1=tanh,2=sigmoid)
        {"low": search_space["learning_rate"][0],"high": search_space["learning_rate"][1]},
        {"low": search_space["epochs"][0],       "high": search_space["epochs"][1]},
    ]

    # 6. PY GAD SETUP
    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=4,
        sol_per_pop=8,
        num_genes=6,
        fitness_func=ga_tuner.fitness_func,
        # BUGFIX: [initial_encoded] * 8 erstellt 8 Referenzen auf dieselbe Liste.
        # Mit list() wird jede Kopie als eigenstaendiges Objekt erstellt.
        initial_population=[list(initial_encoded) for _ in range(8)],
        gene_space=gene_space,
        mutation_percent_genes=20
    )
    starttime = datetime.now()
    print("GA STARTED: " + starttime.strftime('%H:%M:%S'))

    # 7. RUN
    ga.run()

    finishtime = datetime.now()
    print("GA FINISHED: " + finishtime.strftime('%H:%M:%S'))
    dauer = finishtime - starttime
    print("DAUER: " + str(dauer))

    # 8. BESTE LÖSUNG
    solution, fitness, _ = ga.best_solution()

    print("\n========================")
    print("BEST RESULT")
    print("Solution:", solution)
    #print("Fitness:", fitness)
    print("Loss:", -fitness)
    print("========================\n")

    best_params = encoder.decode(solution)

    print("DECODED PARAMS:", best_params)

    # 9. IN TOML SCHREIBEN
    config_loader.config["NeuralNetwork"]["TunedParameters"] = best_params
    config_loader.save(config_path)

    print("SAVED TO TOML")


if __name__ == "__main__":
    main()