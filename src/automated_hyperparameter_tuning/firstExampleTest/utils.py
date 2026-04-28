import torch.nn as knn

#Im Vergleich zu der Lossfunktion haben wir hier mehr :D Aber auch zu Wenig obv.

def get_activation(function):
    if function == "RELU":
        return knn.ReLU()
    elif function == "SIGMOID":
        return knn.Sigmoid()
    else:
        return knn.ReLU()

#Weil es nur n Beispiel ist müssen wir hier obv. nochmehr abfangen als wir derzeit abfangen (nur MSE)

def get_loss(function):
    if function == "MSE":
        return knn.MSELoss()
    else:
        return knn.MSELoss()
