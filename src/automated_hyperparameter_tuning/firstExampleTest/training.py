import torch
from automated_hyperparameter_tuning.firstExampleTest.model import Net
from utils import get_loss

#Trainiert den Bumms halt weil neuronales Netz

def train(layers, lr, epochs, activation_name, X, y):

    model = Net(layers, activation_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss("MSE")

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    return loss.item()