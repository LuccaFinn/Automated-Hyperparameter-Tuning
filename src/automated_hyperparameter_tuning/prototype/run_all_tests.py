import os
import sys
import subprocess
import toml
from pathlib import Path

# Use tomllib from Python standard library (available in 3.11+)
try:
    import tomllib
except ImportError:
    # Fallback to toml if older python is used
    import toml as tomllib

# Config files and their algorithms relative to the project root
CONFIGS = {
    "RandomForest": "resources/configs/gruppe1Config.toml",
    "KNN": "resources/configs/gruppe1Config_KNN.toml",
    "SVM": "resources/configs/gruppe1Config_SVM.toml",
    "LogisticRegression": "resources/configs/gruppe1Config_LogisticRegression.toml",
    "NeuralNetwork": "resources/configs/gruppe1Config_NeuralNetwork.toml"
}

def reset_toml(config_path):
    """
    Resets the TOML configuration: sets status = 'request' and removes the model_tuned section.
    """
    with open(config_path, "rb") as f:
        # tomllib.load requires binary mode
        try:
            data = tomllib.load(f)
        except Exception:
            # Fallback for manual reading if standard parser has issues
            f.seek(0)
            data = toml.loads(f.read().decode('utf-8'))
    
    data["meta"]["status"] = "request"
    if "model_tuned" in data:
        del data["model_tuned"]
        
    with open(config_path, "w") as f:
        toml.dump(data, f)

def run_tuning(python_bin, integration_script, config_path, method, num_steps=2, n_iter=5):
    """
    Runs run_integration.py with the specified settings.
    """
    cmd = [
        python_bin,
        str(integration_script),
        "--config", str(config_path),
        "--method", method,
        "--num_steps", str(num_steps),
        "--n_iter", str(n_iter)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res

def main():
    # Resolve project root dynamically relative to this script's position
    script_path = Path(__file__).resolve()
    project_root = script_path.parent / ".." / ".." / ".."
    project_root = project_root.resolve()
    
    # Path to integration runner script
    integration_script = script_path.parent / "run_integration.py"
    
    # Locate virtual environment python binary
    python_bin = sys.executable  # Use current running python interpreter by default
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_bin = str(venv_python)
        
    results = []
    
    print(f"Projekt-Root: {project_root}")
    print(f"Nutze Python: {python_bin}")
    print("Starte automatisierten Testlauf für alle Algorithmen...\n")
    
    for algo, rel_path in CONFIGS.items():
        config_path = project_root / rel_path
        if not config_path.exists():
            print(f"Fehler: {config_path} existiert nicht!")
            continue
            
        for method in ["grid", "random"]:
            print(f"Teste {algo} mit {method.upper()} Search...")
            
            # Reset before tuning
            reset_toml(config_path)
            
            # Run the integration
            res = run_tuning(python_bin, integration_script, config_path, method)
            
            if res.returncode != 0:
                print(f"  -> FEHLER beim Ausführen von {algo} mit {method}")
                print(res.stderr)
                results.append({
                    "Algorithm": algo,
                    "Method": method,
                    "Status": "FAILED",
                    "Score": "N/A",
                    "Params": "N/A",
                    "Error": res.stderr.strip().split("\n")[-1]
                })
                continue
                
            # Read tuned results from TOML
            with open(config_path, "rb") as f:
                try:
                    tuned_data = tomllib.load(f)
                except Exception:
                    f.seek(0)
                    tuned_data = toml.loads(f.read().decode('utf-8'))
                
            status = tuned_data.get("meta", {}).get("status")
            model_tuned = tuned_data.get("model_tuned", {}).get(algo, {})
            
            if status == "response" and model_tuned:
                score = model_tuned.get("accuracy")
                if score is None:
                    # Look for score under other keys in model_tuned
                    for k, v in model_tuned.items():
                        if k == "accuracy" or isinstance(v, float) and v < 0:
                            score = v
                
                # Extract params (everything except accuracy)
                params = {k: v for k, v in model_tuned.items() if k != "accuracy"}
                
                print(f"  -> ERFOLG: Score = {score}, Best Params = {params}")
                results.append({
                    "Algorithm": algo,
                    "Method": method,
                    "Status": "SUCCESS",
                    "Score": f"{score:.6f}" if isinstance(score, float) else str(score),
                    "Params": str(params),
                    "Error": ""
                })
            else:
                print(f"  -> FEHLER: TOML wurde nicht korrekt aktualisiert. Status: {status}")
                results.append({
                    "Algorithm": algo,
                    "Method": method,
                    "Status": "FAILED",
                    "Score": "N/A",
                    "Params": "N/A",
                    "Error": "TOML status or model_tuned section missing/incorrect"
                })
                
    # Generate markdown report
    markdown_report = "| Algorithmus | Methode | Status | Score (Accuracy / -Loss) | Beste Parameter |\n"
    markdown_report += "| :--- | :--- | :--- | :--- | :--- |\n"
    for r in results:
        status_str = "✅ SUCCESS" if r["Status"] == "SUCCESS" else "❌ FAILED"
        markdown_report += f"| {r['Algorithm']} | {r['Method'].upper()} | {status_str} | {r['Score']} | `{r['Params']}` |\n"
        
    print("\n=== TEST ERGEBNISSE (MARKDOWN) ===\n")
    print(markdown_report)
    
    # Save the markdown report locally to the project
    report_file = project_root / "walkthrough_test_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Testergebnisse Hyperparameter-Tuning\n\n")
        f.write(markdown_report)
        f.write("\n\n*Dieser Bericht wurde automatisch generiert.*")
    print(f"\nBericht gespeichert unter: {report_file}")

if __name__ == "__main__":
    main()
