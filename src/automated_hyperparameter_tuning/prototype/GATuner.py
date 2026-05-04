from NeuralNet import NeuralNet

#GeneticAlgorithm, dafur der Name, also damit arbeitet pyGAd halt wa?
class GATuner:
    def __init__(self, encoder, trainer, data, input_size):
        self.encoder = encoder
        self.trainer = trainer
        self.data = data
        self.input_size = input_size

    def fitness_func(self, ga, solution, idx):
        params = self.encoder.decode(solution)
        #Gleicher Bumms wie eben beim Trainer - hier ist Herr Wucke Beauftragter
        params["learning_rate"] = abs(params["learning_rate"])

        model = NeuralNet(params, self.input_size)
        score = self.trainer.train(model, self.data, params)

        return score