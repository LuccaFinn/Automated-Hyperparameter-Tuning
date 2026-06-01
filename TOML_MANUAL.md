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
      * `input_start` Index der ersten Spalte, die als Feature verwendet wird (inklusive). Zaehlung beginnt bei 0.
      * `input_end` Index der letzten Spalte, die als Feature verwendet wird (inklusive).
      * `target_column` Index der Spalte, die als Zielvariable verwendet wird.
      * `test_size` Anteil der Daten, der fuer die Validierung reserviert wird. Wert zwischen 0.0 und 1.0.
      * `random_state` Seed fuer die zufaellige Aufteilung der Daten. Sorgt fuer Reproduzierbarkeit.
---

## 3. [InitialParameters]

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
