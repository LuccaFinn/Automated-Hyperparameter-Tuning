# Automated Hyperparameter Tuning
Dieses Projekt ist im Rahmen des Integrationsprojektes „Evaluierung unterschiedlicher Machine-Learning-Ansätze für datenbasierte Modellgeneration" entstanden. Hierbei gilt es auf Basis der systematischen Untersuchung von Hyperparametern geeignete Strategien zur automatisierten Optimierung zu entwickeln. Der Fokus liegt auf der Analyse und dem Vergleich von Verfahren wie [Randomized Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#random_search) und [Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) sowie heuristischen Ansätzen, beispielsweise [evolutionären Algorithmen](https://en.wikipedia.org/wiki/Evolutionary_algorithm).

## Installation & Voraussetzungen & Externe Abhängigkeiten
Für die Nutzung des Tuners wird PyCharm empfohlen. Die benötigten Packages sind der `requirements.txt` zu entnehmen,
können jedoch bei der Nutzung von PyCharm auch automatisch installiert werden. 

## Mögliche Algorithmen
In diesem Projekt stehen insgesamt sechs verschiedene Algorithmen zur Verfügung dessen Hyperparameter getuned werden können.
Zu diesen zählen die folglich genannten:
* **k-Nearest Neighbors (knn)**
* **Logistische Regression (logistic_regression)**
* **Neuronale Netze (neural_network)**
* **Support Vector Machine (svm)**
* **XGBoost (xgboost)**

## Konfiguration per TOML
Eine Aufgabe des Projektes bestand darin eine Schnittstelle zu entwerfen, um Daten zwischen Gruppe 1 und Gruppe 3 auszutauschen.
Eine `.toml` Datei stellt dabei diese Schnittstelle dar. Wie diese aussieht wird im Folgenden dargestellt. Diese
Darstellung, mit den angegebenen Parametern, gilt jedoch ausschließlich für den Algorithmus "k-Nearest Neighbors". 

**Die Konfigurationsdatei (`config_knn`):**
```
[Model]
algorithm = "knn"
task = "classification"

[Data]
input_start = 0
input_end = 6
target_column = 7
test_size = 0.2
random_state = 42

[KNN.InitialParameters]
n_neighbors = 5
weights = "uniform"
metric = "euclidean"

[KNN.SearchSpace]
n_neighbors = [ 1, 50,]
weights = [ "uniform", "distance",]
metric = [ "euclidean", "manhattan", "minkowski",]

[KNN.TunedParameters]
n_neighbors = 5
weights = "uniform"
metric = "euclidean"
```

> **Hinweis:** Eine vollständige Liste und detaillierte Erklärung aller möglichen TOML-Parameter findet sich hier: [TOML Konfigurations-Handbuch](TOML_MANUAL.md)
## Beispieldurchlauf
Um die Hyperparameter eines beliebigen, obig genannten Algorithmus mit Hilfe von genetischen Algorithmen zu tunen,
steht in dem Projekt unter `src/automated_hyperparameter_tuning/prototype` pro Algorithmus eine `main.py` zur Verfügung. Die obige genannten `.toml` Dateien
sind ebenfalls in dem Projekt unter `src/configs/` zur Verfügung. Um nun diese `main.py` nun ausführen zu können muss ggf. die `.toml` auf das gewünschte Problem angepasst werden.
Des Weiteren muss in dem `ConfigLoader.py` der Pfad zur CSV Datei, welche in `resources/configs` zu finden sind, angepasst werden.

## Suchmethoden

Pro Algorithmus stehen neben dem genetischen Algorithmus zwei weitere Suchmethoden zur Verfügung, um die Ergebnisse vergleichen zu können:

| Methode | Beschreibung | Datei |
|---|---|---|
| **Genetischer Algorithmus** | Evolutionäre Suche über den SearchSpace. Ergebnis landet in `TunedParameters`. | `main_<algorithmus>.py` |
| **Grid Search** | Bruteforce Suche nach den besten Werten. Ergebnis landet in `GridSearchParameters`. | `gridsearch_<algorithmus>.py` |
| **Random Search** | Zufällige Stichproben aus dem SearchSpace. Anzahl der Iterationen ist per TOML konfigurierbar. Ergebnis landet in `RandomSearchParameters`. | `randomsearch_<algorithmus>.py` |

Nach dem Durchlauf aller drei Methoden können die Ergebnisse direkt in der `.toml` verglichen werden und durch `verfication.py` verifiziert werden.

## Projektstruktur

```
Automated-Hyperparameter-Tuning/
├── resources/
│   ├── configs/          # TOML-Konfigurationsdateien pro Algorithmus
│   └── data/             # CSV-Datendateien
└── src/
    └── automated_hyperparameter_tuning/
        └── prototype/
            ├── ConfigLoader.py
            ├── EncoderDecoder.py
            ├── SKLearnEncoderDecoder.py
            ├── Trainer.py
            ├── SKLearnTrainer.py
            ├── GATuner.py
            ├── SKLearnGATuner.py
            ├── NeuralNet.py
            ├── verification.py
            ├── main_neural_network.py
            ├── main_svm.py
            ├── main_knn.py
            ├── main_logistic_regression.py
            ├── main_xgboost.py
            ├── gridsearch_neural_network.py
            ├── gridsearch_svm.py
            ├── randomsearch_neural_network.py
            ├── randomsearch_svm.py
            └── ConfigConfigurator.py         # Veraltete Datei, die nicht benutzt wurde
```

## Autor*innen

[Björn](https://github.com/bjzrn) | [Katharina](https://github.com/kasarahkoe-dotcom) | [Lucca](https://github.com/LuccaFinn) | [Jonas](https://github.com/jonasc4)

Privates Projekt - Alle Rechte vorbehalten.
---
*Zuletzt aktualisiert: Juni 2026*
