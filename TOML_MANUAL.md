# TOML Konfigurations-Handbuch

Die `config.toml` ist die Schnittstelle zwischen Gruppe 1 und Gruppe 3. Sie erlaubt es, den gewünschten Algorithmus, die gewünschten Hyperparameter samt Search-Space zu definieren, ohne eine Zeile des Codes ändern zu müssen.

Dieses Handbuch erklärt alle verfügbaren Blöcke und Parameter.

##  1. [Model]

* `Algorithm` Bestimmt von welchem Algorithmus die Hyperparameter getuned werden sollen?
    * k-Nearest Neighbors **(knn)**
    * Logistische Regression **(logistic_regression)**
    * Neuronale Netze **(nn)**
    * Support Vector Machine **(svm)**
    * XGBoost **(xgboost)**
* `Task` Ist das Problem ein Klassifikations- oder Regressionsproblem?
    * Klassifikation **(classification)**
    * Regression **(regression)**
---

## 2. [Data]

* Definiert den Pfad zu den Eingabedaten sowie die Aufteilung in Features und Zielvariable.
      * `input_start` Index der ersten Spalte, die als Feature verwendet wird (inklusive). Zählung beginnt bei 0.
      * `input_end` Index der letzten Spalte, die als Feature verwendet wird (inklusive).
      * `target_column` Index der Spalte, die als Zielvariable verwendet wird.
      * `test_size` Anteil der Daten, der für die Validierung reserviert wird. Wert zwischen 0.0 und 1.0.
      * `random_state` Seed für die zufällige Aufteilung der Daten. Sorgt für Reproduzierbarkeit.
---

## 3. [InitialParameters]
Definiert den Startpunkt für den genetischen Algorithmus. Das erste Individuum der Population wird mit diesen Werten initialisiert. Alle weiteren Individuen sind Kopien davon, die dann durch Mutation variiert werden.

Die verfügbaren Parameter hängen vom gewählten Algorithmus ab:

**neural_network**

| Parameter | Beschreibung | Beispielwert |
|---|---|---|
| `layer1` | Anzahl Neuronen in der ersten versteckten Schicht | `32` |
| `layer2` | Anzahl Neuronen in der zweiten versteckten Schicht | `16` |
| `layer3` | Anzahl Neuronen in der dritten versteckten Schicht | `8` |
| `activation` | Aktivierungsfunktion (`relu`, `tanh`, `sigmoid`) | `"relu"` |
| `learning_rate` | Lernrate | `0.001` |
| `epochs` | Maximale Anzahl Trainingsepochen | `100` |
| `loss_function` | Verlustfunktion (`mse`, `mae` für Regression; `mse`, `bce` für Klassifikation) | `"mse"` |

**svm**

| Parameter | Beschreibung | Beispielwert |
|---|---|---|
| `C` | Regularisierungsparameter. Größere Werte = weniger Regularisierung | `1.0` |
| `kernel` | Kernelfunktion (`linear`, `rbf`, `poly`, `sigmoid`) | `"rbf"` |
| `gamma` | Einflussreichweite eines Trainingspunktes (`scale`, `auto`) | `"scale"` |

**knn**

| Parameter | Beschreibung | Beispielwert |
|---|---|---|
| `n_neighbors` | Anzahl der nächsten Nachbarn | `5` |
| `weights` | Gewichtung der Nachbarn (`uniform`, `distance`) | `"uniform"` |
| `metric` | Distanzmetrik (`euclidean`, `manhattan`, `minkowski`) | `"euclidean"` |

**logistic_regression**

| Parameter | Beschreibung | Beispielwert |
|---|---|---|
| `C` | Regularisierungsparameter. Größere Werte = weniger Regularisierung | `1.0` |
| `max_iter` | Maximale Anzahl Iterationen bis zur Konvergenz | `100` |
| `solver` | Optimierungsalgorithmus (`lbfgs`, `saga`) | `"lbfgs"` |

**xgboost**

| Parameter | Beschreibung | Beispielwert |
|---|---|---|
| `n_estimators` | Anzahl der Entscheidungsbäume | `100` |
| `max_depth` | Maximale Tiefe eines Baumes | `3` |
| `learning_rate` | Lernrate (auch Shrinkage genannt) | `0.1` |
| `subsample` | Anteil der Trainingsdaten pro Baum. Wert zwischen 0.0 und 1.0 | `1.0` |


---

## 4. [SearchSpace]

Definiert die Grenzen, innerhalb derer der genetische Algorithmus nach optimalen Hyperparametern sucht. Für jeden Parameter aus InitialParameters wird hier ein Wertebereich angegeben.
Beispiel für ein neuronales Netz:

```[NeuralNetwork.SearchSpace]
layer1 = [1, 128]
layer2 = [1, 256]
layer3 = [1, 64]
activation = ["relu", "tanh", "sigmoid"]
learning_rate = [1e-5, 1.0]
epochs = [10, 500]
loss_function = ["mse", "mae"]```

> **Hinweis:** Hinweis: Je größer der SearchSpace, desto länger benötigt der genetische Algorithmus. Es empfiehlt sich, den SearchSpace auf plausible Wertebereiche einzuschränken.
---

## 5. [TunedParameters]
Wird automatisch vom Programm befüllt, nachdem der genetische Algorithmus abgeschlossen hat. Dieser Block muss in der `.toml` vorhanden sein, sollte aber nicht manuell verändert werden.
Nach dem Tuning enthält dieser Bereich die besten gefundenen Hyperparameter und kann zur Verifikation und zum direkten Aufbau des Modells verwendet werden.
