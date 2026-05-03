from NeuralNet import NeuralNet  # bleibt gleich, wird durch main.py sys.path geloest


class GATuner:
    def __init__(self, encoder, trainer, data, input_size):
        self.encoder = encoder
        self.trainer = trainer
        self.data = data
        self.input_size = input_size

    def fitness_func(self, ga, solution, idx):
        params = self.encoder.decode(solution)
        params["learning_rate"] = abs(params["learning_rate"])

        model = NeuralNet(params, self.input_size)
        score = self.trainer.train(model, self.data, params)

        return score