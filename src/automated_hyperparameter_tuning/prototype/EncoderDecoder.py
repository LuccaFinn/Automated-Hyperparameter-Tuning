class EncoderDecoder:
    def __init__(self, search_space):
        #Die Dinger stehen halt zur Auswahl und brauchen halt ne Nummer damit opygad was damit anfangen kann
        self.activation_map = {
            0: "relu",
            1: "tanh",
            2: "sigmoid"
        }
        self.reverse_map = {v: k for k, v in self.activation_map.items()}

        self.loss_map = {
            0: "mse",
            1: "bce"
        }
        self.reverse_loss_map = {v: k for k, v in self.loss_map.items()}

    #der kodiert sich das intern lustig (siehe ausgabe am ende) sodass man so nen encoder und decoder braucht
    def encode(self, params):
        return [
            params["layer1"],
            params["layer2"],
            params["layer3"],
            self.reverse_map[params["activation"]],
            params["learning_rate"],
            params["epochs"],
            self.reverse_loss_map[params["loss_function"]]
        ]

    def decode(self, solution):
        return {
            "layer1": int(solution[0]),
            "layer2": int(solution[1]),
            "layer3": int(solution[2]),
            "activation": self.activation_map[int(solution[3])],
            "learning_rate": float(solution[4]),
            "epochs": int(solution[5]),
            "loss_function": self.loss_map[int(solution[6])]
        }