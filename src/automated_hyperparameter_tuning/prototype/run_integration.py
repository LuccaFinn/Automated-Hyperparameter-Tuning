import os
import sys
import argparse
import toml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add current directory to python path
sys.path.insert(0, os.path.dirname(__file__))

from SKLearnTrainer import SKLearnTrainer
from Trainer import Trainer
from GridSearchTuner import GridSearchTuner
from RandomizedSearchTuner import RandomizedSearchTuner

def main():
    parser = argparse.ArgumentParser(description="AutoML Hyperparameter Tuning Integration Interface")
    parser.add_argument(
        "--config", 
        type=str, 
        default="resources/configs/gruppe1Config.toml",
        help="Path to the TOML configuration file from Group 1"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Hyperparameter optimization method (grid or random)"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="Number of iterations for randomized search"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps for numeric ranges in Grid Search"
    )
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        # Try relative to script parent directory
        config_path = Path(os.path.dirname(__file__)).resolve() / ".." / ".." / ".." / args.config
        config_path = config_path.resolve()
        
    if not config_path.exists():
        print(f"Fehler: Konfigurationsdatei nicht gefunden unter {args.config}")
        sys.exit(1)
        
    print(f"Lade Konfiguration: {config_path}")
    import tomllib
    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)
    
    meta = toml_data.get("meta", {})
    
    # Check if this is the new format (has search_spaces)
    is_new_format = "search_spaces" in toml_data
    
    if is_new_format:
        print("Erkenne neues TOML-Format von Gruppe 1...")
        dataset_path_raw = meta.get("dataset_path") or meta.get("data_path")
        target_column = meta.get("target_column")
        selected_architectures = toml_data.get("selected_architectures", {})
        algorithms = list(selected_architectures.values())
        search_spaces_all = toml_data.get("search_spaces", {})
    else:
        print("Erkenne altes TOML-Format von Gruppe 1...")
        dataset_path_raw = meta.get("data_path")
        target_column = None
        algorithms = [meta.get("selected_algorithm")]
        search_spaces_all = {algorithms[0]: toml_data.get("model_to_tune", {}).get(algorithms[0], {})}
        
    if not dataset_path_raw or not algorithms or not all(algorithms):
        print("Fehler: Fehlende Metadaten (dataset_path/data_path, selected_algorithm/selected_architectures) in der TOML-Konfiguration.")
        sys.exit(1)
        
    print(f"Ausgewählte Algorithmen: {algorithms}")
    
    # Resolve CSV Path
    csv_path = Path(dataset_path_raw).resolve()
    if not csv_path.exists():
        # Check relative to config path directory
        csv_path = config_path.parent / dataset_path_raw
        csv_path = csv_path.resolve()
        
    if not csv_path.exists():
        # Look in resources/data/
        script_dir = Path(os.path.dirname(__file__)).resolve()
        data_dir = (script_dir / ".." / ".." / ".." / "resources" / "data").resolve()
        target_name = os.path.basename(dataset_path_raw).lower()
        
        if data_dir.exists():
            for f in data_dir.iterdir():
                if f.name.lower() == target_name:
                    csv_path = f
                    break
                    
    if not csv_path.exists():
        # Check in local resources/data/ folder
        data_dir_local = Path("resources/data").resolve()
        target_name = os.path.basename(dataset_path_raw).lower()
        if data_dir_local.exists():
            for f in data_dir_local.iterdir():
                if f.name.lower() == target_name:
                    csv_path = f
                    break

    if not csv_path.exists():
        # Fallback to exampleCSVBanana.csv for testing
        fallback_path = Path("resources/data/exampleCSVBanana.csv").resolve()
        if fallback_path.exists():
            print(f"Warnung: CSV '{dataset_path_raw}' nicht gefunden. Verwende Fallback '{fallback_path.name}' für Testzwecke.")
            csv_path = fallback_path
        else:
            print(f"Fehler: CSV-Datei '{dataset_path_raw}' nicht gefunden.")
            sys.exit(1)
            
    print(f"Lade Datensatz von: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract features and target
    if target_column and target_column in df.columns:
        print(f"Verwende Zielspalte: '{target_column}'")
        X_df = df.drop(columns=[target_column])
        y_raw = df[target_column].values
    else:
        # Check case insensitively
        found_target = None
        if target_column:
            for col in df.columns:
                if col.lower() == target_column.lower():
                    found_target = col
                    break
        if found_target:
            print(f"Verwende Zielspalte (case-insensitive Match): '{found_target}'")
            X_df = df.drop(columns=[found_target])
            y_raw = df[found_target].values
        else:
            # Default to last column
            target_col_default = df.columns[-1]
            print(f"Warnung: Zielspalte '{target_column}' nicht im Datensatz gefunden. Nutze Standard (letzte Spalte): '{target_col_default}'")
            X_df = df.iloc[:, :-1]
            y_raw = df.iloc[:, -1].values
            
    X = X_df.values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(float)
    
    # We will run tuning for each algorithm and collect results
    tuned_hyperparameters = {}
    
    for selected_algorithm in algorithms:
        if not selected_algorithm:
            continue
            
        print(f"\n--- Starte Tuning für {selected_algorithm} ---")
        search_space = search_spaces_all.get(selected_algorithm, {})
        if not search_space:
            print(f"Warnung: Kein Suchraum für '{selected_algorithm}' gefunden. Überspringe...")
            continue
            
        print(f"Suchraum für {selected_algorithm}: {search_space}")
        
        # Prepare train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        data = (
            torch_tensor_if_needed(X_train, selected_algorithm),
            torch_tensor_if_needed(X_val, selected_algorithm),
            torch_tensor_if_needed(y_train, selected_algorithm).reshape(-1, 1) if selected_algorithm in ["NeuralNetwork", "neural_network"] else y_train,
            torch_tensor_if_needed(y_val, selected_algorithm).reshape(-1, 1) if selected_algorithm in ["NeuralNetwork", "neural_network"] else y_val
        )
        
        # Configure Trainer & Tuner
        if selected_algorithm in ["NeuralNetwork", "neural_network"]:
            trainer = Trainer(early_stopping=False)
            input_size = X_train.shape[1]
            if args.method == "grid":
                tuner = GridSearchTuner(
                    algorithm="neural_network",
                    trainer=trainer,
                    data=data,
                    input_size=input_size,
                    num_steps=args.num_steps
                )
            else:
                tuner = RandomizedSearchTuner(
                    algorithm="neural_network",
                    trainer=trainer,
                    data=data,
                    input_size=input_size,
                    n_iter=args.n_iter
                )
        else:
            # Scikit-Learn
            trainer = SKLearnTrainer()
            if args.method == "grid":
                tuner = GridSearchTuner(
                    algorithm=selected_algorithm,
                    trainer=trainer,
                    data=data,
                    num_steps=args.num_steps
                )
            else:
                tuner = RandomizedSearchTuner(
                    algorithm=selected_algorithm,
                    trainer=trainer,
                    data=data,
                    n_iter=args.n_iter
                )
                
        starttime = datetime.now()
        print(f"Optimierung gestartet ({args.method.upper()}) um: " + starttime.strftime('%H:%M:%S'))
        
        best_params, best_score, best_model = tuner.tune(search_space)
        
        finishtime = datetime.now()
        print("Optimierung beendet um: " + finishtime.strftime('%H:%M:%S'))
        print("Dauer insgesamt: " + str(finishtime - starttime))
        
        print(f"Bestes Ergebnis für {selected_algorithm}: {best_score} mit Parametern: {best_params}")
        
        # Clean numeric types for TOML output
        best_params_clean = {}
        for k, v in best_params.items():
            if isinstance(v, (np.integer, np.int64)):
                best_params_clean[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                best_params_clean[k] = float(v)
            else:
                best_params_clean[k] = v
                
        # Also include accuracy/score inside response
        best_params_clean["accuracy"] = float(best_score)
        
        tuned_hyperparameters[selected_algorithm] = best_params_clean

    if is_new_format:
        # Write to hp_response.toml in the same folder
        output_path = config_path.parent / "hp_response.toml"
        print(f"\nSchreibe Ergebnisse im neuen Format nach: {output_path}")
        
        # Format inline tables manually to match the exact group 1 response format
        response_lines = ["[tuned_hyperparameters]"]
        for algo, params in tuned_hyperparameters.items():
            param_pairs = []
            for k, v in params.items():
                if isinstance(v, str):
                    param_pairs.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    param_pairs.append(f'{k} = {"true" if v else "false"}')
                elif v is None:
                    param_pairs.append(f'{k} = "None"')
                else:
                    if isinstance(v, float):
                        param_pairs.append(f'{k} = {v:.6g}')
                    else:
                        param_pairs.append(f'{k} = {v}')
            param_str = ", ".join(param_pairs)
            response_lines.append(f"{algo} = {{ {param_str} }}")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(response_lines) + "\n")
            
        print(f"Erfolgreich geantwortet. Datei erstellt: {output_path}")
    else:
        # Write back to same TOML file
        toml_data["meta"]["status"] = "response"
        if "model_tuned" not in toml_data:
            toml_data["model_tuned"] = {}
            
        for algo, best_params_clean in tuned_hyperparameters.items():
            score = best_params_clean.pop("accuracy", 0.0)
            toml_data["model_tuned"][algo] = best_params_clean
            toml_data["model_tuned"][algo]["accuracy"] = score
            
        with open(config_path, "w") as f:
            toml.dump(toml_data, f)
            
        print(f"Ergebnisse zurückgeschrieben und gespeichert unter: {config_path}")

def torch_tensor_if_needed(arr, algorithm):
    if algorithm in ["NeuralNetwork", "neural_network"]:
        import torch
        return torch.tensor(arr, dtype=torch.float32)
    return arr

if __name__ == "__main__":
    main()
