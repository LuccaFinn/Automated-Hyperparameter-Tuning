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
