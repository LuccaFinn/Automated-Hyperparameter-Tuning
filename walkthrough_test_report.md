# Testergebnisse Hyperparameter-Tuning

| Algorithmus | Methode | Status | Score (Accuracy / -Loss) | Beste Parameter |
| :--- | :--- | :--- | :--- | :--- |
| RandomForest | GRID | ✅ SUCCESS | 0.976250 | `{'n_estimators': 200, 'max_depth': 20}` |
| RandomForest | RANDOM | ✅ SUCCESS | 0.976250 | `{'n_estimators': 200, 'max_depth': 'None'}` |
| KNN | GRID | ✅ SUCCESS | 0.980625 | `{'n_neighbors': 20, 'weights': 'uniform', 'metric': 'manhattan'}` |
| KNN | RANDOM | ✅ SUCCESS | 0.980625 | `{'n_neighbors': 30, 'weights': 'distance', 'metric': 'manhattan'}` |
| SVM | GRID | ✅ SUCCESS | 0.984375 | `{'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'}` |
| SVM | RANDOM | ✅ SUCCESS | 0.984375 | `{'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'}` |
| LogisticRegression | GRID | ✅ SUCCESS | 0.879375 | `{'C': 0.1, 'max_iter': 100, 'solver': 'lbfgs', 'penalty': 'l2'}` |
| LogisticRegression | RANDOM | ✅ SUCCESS | 0.879375 | `{'C': 1.0, 'max_iter': 127, 'solver': 'lbfgs', 'penalty': 'l2'}` |
| NeuralNetwork | GRID | ✅ SUCCESS | -0.052590 | `{'layer1': 50, 'layer2': 10, 'layer3': 1, 'activation': 'tanh', 'learning_rate': 0.01, 'epochs': 20, 'loss_function': 'mse'}` |
| NeuralNetwork | RANDOM | ✅ SUCCESS | -0.077952 | `{'layer1': 33, 'layer2': 31, 'layer3': 1, 'activation': 'relu', 'learning_rate': 0.007841175250812004, 'epochs': 17, 'loss_function': 'mse'}` |


*Dieser Bericht wurde automatisch generiert.*