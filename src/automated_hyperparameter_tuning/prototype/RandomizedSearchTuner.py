import random
import numpy as np

class RandomizedSearchTuner:
    def __init__(self, algorithm, trainer, data, input_size=None, n_iter=10):
        """
        Randomized Search Tuner for Hyperparameter Optimization.
        
        Args:
            algorithm (str): Name of the algorithm (e.g. 'knn', 'svm', 'neural_network').
            trainer: Trainer instance (Trainer or SKLearnTrainer).
            data (tuple): Training and validation data tuple (X_train, X_val, y_train, y_val).
            input_size (int, optional): Input feature size for neural networks.
            n_iter (int): Number of random parameter combinations to try.
        """
        self.algorithm = algorithm
        self.trainer = trainer
        self.data = data
        self.input_size = input_size
        self.n_iter = n_iter

    def sample_parameters(self, search_space):
        """
        Randomly samples one parameter combination from the search space.
        """
        params = {}
        for key, value in search_space.items():
            if not isinstance(value, list):
                params[key] = value
                continue
            
            if len(value) == 0:
                continue
                
            # If it's a categorical list or has more than 2 elements
            # (e.g. ["relu", "tanh", "sigmoid"] or [100, 200, 500])
            if len(value) > 2 or any(isinstance(x, str) for x in value):
                params[key] = random.choice(value)
                continue
            
            # If it has exactly 2 elements and they are numbers, treat it as [min, max] range
            if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
                low, high = value[0], value[1]
                # If both are integers
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = random.randint(low, high)
                else:
                    # Floats, e.g. learning rate
                    # If high/low spans more than 2 orders of magnitude, use log-uniform sampling
                    if low > 0 and high / low >= 100:
                        log_low = np.log10(low)
                        log_high = np.log10(high)
                        params[key] = float(10 ** random.uniform(log_low, log_high))
                    else:
                        params[key] = float(random.uniform(low, high))
            else:
                params[key] = random.choice(value)
                
        return params

    def make_hashable(self, params):
        """
        Converts parameter dict into a hashable tuple of sorted key-value pairs for duplicate checking.
        """
        return tuple(sorted((k, str(v)) for k, v in params.items()))

    def tune(self, search_space):
        """
        Runs the randomized search over n_iter iterations.
        
        Returns:
            best_params (dict): Dict of the best parameters found.
            best_score (float): Score (accuracy or negative loss) of the best parameters.
            best_model: Trained model with the best parameters.
        """
        best_score = -float('inf')
        best_params = None
        best_model = None
        
        seen_configs = set()
        
        print(f"Starte Randomized Search mit {self.n_iter} Iterationen...")
        
        for idx in range(self.n_iter):
            # Try to get a unique configuration, up to 100 retries
            params = None
            for _ in range(100):
                candidate = self.sample_parameters(search_space)
                candidate_hash = self.make_hashable(candidate)
                if candidate_hash not in seen_configs:
                    params = candidate
                    seen_configs.add(candidate_hash)
                    break
            
            if params is None:
                # If we couldn't find a new unique config, we just use the last sampled candidate
                params = candidate
                
            # Make sure specific types are correct
            for int_key in ["layer1", "layer2", "layer3", "epochs", "n_neighbors", "max_iter"]:
                if int_key in params:
                    params[int_key] = int(params[int_key])
            
            if self.algorithm in ["neural_network", "neural_net"]:
                from NeuralNet import NeuralNet
                model = NeuralNet(params, self.input_size)
                score = self.trainer.train(model, self.data, params)
                model_to_save = model
            else:
                score, model = self.trainer.train(self.algorithm, self.data, params)
                model_to_save = model
                
            print(f"Iteration {idx+1}/{self.n_iter}: {params} -> Score (Fitness): {score:.6f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model_to_save
                
        return best_params, best_score, best_model
