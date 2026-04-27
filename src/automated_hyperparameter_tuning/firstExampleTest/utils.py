import torch.nn as nn

#Im Vergleich zu der Lossfunktion haben wir hier mehr :D Aber auch zu Wenig obv.

def get_activation(function):
    if function == "RELU":
        return nn.ReLU()
    elif function == "SIGMOID":
        return nn.Sigmoid()
    else:
        return nn.ReLU()

#Weil es nur n Beispiel ist müssen wir hier obv. nochmehr abfangen als wir derzeit abfangen (nur MSE)

def get_loss(function):
    if function == "MSE":
        return nn.MSELoss()
    else:
        return nn.MSELoss()