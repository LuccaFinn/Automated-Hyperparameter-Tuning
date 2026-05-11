class SKLearnGATuner:
    def __init__(self, algorithm, encoder, trainer, data):
        self.algorithm = algorithm
        self.encoder   = encoder
        self.trainer   = trainer
        self.data      = data

    def fitness_func(self, ga, solution, idx):
        params = self.encoder.decode(solution)
        score, _ = self.trainer.train(self.algorithm, self.data, params)
        return score
