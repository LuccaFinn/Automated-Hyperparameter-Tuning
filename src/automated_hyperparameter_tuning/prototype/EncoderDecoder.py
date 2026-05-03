class EncoderDecoder:
    def __init__(self, search_space):
        self.activation_map = {
            0: "relu",
            1: "tanh",
            2: "sigmoid"
        }
        self.reverse_map = {v: k for k, v in self.activation_map.items()}

    def encode(self, params):
        return [
            params["layer1"],
            params["layer2"],
            params["layer3"],
            self.reverse_map[params["activation"]],
            params["learning_rate"],
            params["epochs"]
        ]

    def decode(self, solution):
        return {
            "layer1": int(solution[0]),
            "layer2": int(solution[1]),
            "layer3": int(solution[2]),
            "activation": self.activation_map[int(solution[3])],
            "learning_rate": float(solution[4]),
            "epochs": int(solution[5])
        }