import os
import sys
import argparse
from datetime import datetime

# Add the directory containing this script to python path
sys.path.insert(0, os.path.dirname(__file__))

from ConfigLoader import ConfigLoader
from SKLearnTrainer import SKLearnTrainer
from Trainer import Trainer
from RandomizedSearchTuner import RandomizedSearchTuner

def main():
    parser = argparse.ArgumentParser(description="Randomized Search Hyperparameter Tuning")
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="knn", 
        choices=["knn", "svm", "logistic_regression", "neural_network"],
        help="Algorithm to tune (knn, svm, logistic_regression, neural_network)"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="Number of random parameter combinations to try"
    )
    args = parser.parse_args()
    
    algorithm = args.algorithm
    print(f"==================================================")
    print(f"Randomized Search für {algorithm.upper()} gestartet")
    print(f"==================================================")
    
    # Map algorithm to config filename
    config_names = {
        "knn": "config_knn.toml",
        "svm": "config_svm.toml",
        "logistic_regression": "config_logistic_regression.toml",
        "neural_network": "configForPrototype.toml"
    }
    
    config_file = config_names.get(algorithm)
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "resources", "configs", config_file
    )
    
    config_loader = ConfigLoader(config_path)
    
    # Fallback checking for CSV file in resources/data if default doesn't exist
    if not config_loader.absolute_csv_path.exists():
        fallback_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "resources", "data", "exampleCSVBanana.csv"
        )
        import pathlib
        if os.path.exists(fallback_path):
            config_loader.absolute_csv_path = pathlib.Path(fallback_path).resolve()
            print(f"Nutze Fallback-CSV-Pfad: {config_loader.absolute_csv_path}")

    # Load data
    X_train, X_val, y_train, y_val = config_loader.load_data()
    data = (X_train, X_val, y_train, y_val)
    
    # Tune
    if algorithm == "neural_network":
        search_space = config_loader.config["NeuralNetwork"]["SearchSpace"]
        trainer = Trainer(early_stopping=False)
        input_size = X_train.shape[1]
        tuner = RandomizedSearchTuner(
            algorithm=algorithm,
            trainer=trainer,
            data=data,
            input_size=input_size,
            n_iter=args.n_iter
        )
    else:
        # Sklearn models
        toml_key = "KNN" if algorithm == "knn" else ("SVM" if algorithm == "svm" else "LogisticRegression")
        search_space = config_loader.config[toml_key]["SearchSpace"]
        trainer = SKLearnTrainer()
        tuner = RandomizedSearchTuner(
            algorithm=algorithm,
            trainer=trainer,
            data=data,
            n_iter=args.n_iter
        )
        
    starttime = datetime.now()
    print("Randomized Search gestartet um: " + starttime.strftime('%H:%M:%S'))
    
    best_params, best_score, best_model = tuner.tune(search_space)
    
    finishtime = datetime.now()
    print("Randomized Search beendet um: " + finishtime.strftime('%H:%M:%S'))
    print("Dauer insgesamt: " + str(finishtime - starttime))
    
    print("\n------------------")
    print("Bestes Ergebnis")
    if algorithm == "neural_network":
        print("Val Loss (MSE/BCE):", -best_score)
    else:
        print("Accuracy:", best_score)
    print("Parameter:", best_params)
    print("------------------\n")
    
    # Save tuned parameters to TOML
    if algorithm == "neural_network":
        config_loader.config["NeuralNetwork"]["TunedParameters"] = best_params
    else:
        toml_key = "KNN" if algorithm == "knn" else ("SVM" if algorithm == "svm" else "LogisticRegression")
        config_loader.config[toml_key]["TunedParameters"] = best_params
        
    config_loader.save(config_path)
    print("Optimierte Parameter zurück in TOML geschrieben.")

if __name__ == "__main__":
    main()
