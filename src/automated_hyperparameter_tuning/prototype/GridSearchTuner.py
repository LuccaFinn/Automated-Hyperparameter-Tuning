import itertools
import numpy as np

class GridSearchTuner:
    def __init__(self, algorithm, trainer, data, input_size=None, num_steps=5):
        """
        Grid Search Tuner for Hyperparameter Optimization.
        
        Args:
            algorithm (str): Name of the algorithm (e.g. 'knn', 'svm', 'neural_network').
            trainer: Trainer instance (Trainer or SKLearnTrainer).
            data (tuple): Training and validation data tuple (X_train, X_val, y_train, y_val).
            input_size (int, optional): Input feature size for neural networks.
            num_steps (int): Number of steps to generate for numeric ranges in Grid Search.
        """
        self.algorithm = algorithm
        self.trainer = trainer
        self.data = data
        self.input_size = input_size
        self.num_steps = num_steps

    def generate_grid(self, search_space):
        """
        Generates a discrete parameter grid from the given search space.
        """
        grid = {}
        for key, value in search_space.items():
            if not isinstance(value, list):
                # Single value parameter
                grid[key] = [value]
                continue
            
            if len(value) == 0:
                grid[key] = []
                continue
                
            # If it's a categorical list or has more than 2 elements
            # (e.g. ["relu", "tanh", "sigmoid"] or [100, 200, 500])
            if len(value) > 2 or any(isinstance(x, str) for x in value):
                grid[key] = value
                continue
            
            # If it has exactly 2 elements and they are numbers, treat it as [min, max] range
            if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
                low, high = value[0], value[1]
                # If both are integers
                if isinstance(low, int) and isinstance(high, int):
                    vals = np.linspace(low, high, num=min(self.num_steps, high - low + 1))
                    grid[key] = sorted(list(set(int(round(v)) for v in vals)))
                else:
                    # Floats, e.g. learning rate
                    # If high/low spans more than 2 orders of magnitude, use log-spacing
                    if low > 0 and high / low >= 100:
                        vals = np.logspace(np.log10(low), np.log10(high), num=self.num_steps)
                    else:
                        vals = np.linspace(low, high, num=self.num_steps)
                    grid[key] = sorted(list(set(float(v) for v in vals)))
            else:
                grid[key] = value
                
        return grid

    def tune(self, search_space):
        """
        Runs the grid search over the combinations generated from search_space.
        
        Returns:
            best_params (dict): Dict of the best parameters found.
            best_score (float): Score (accuracy or negative loss) of the best parameters.
            best_model: Trained model with the best parameters.
        """
        grid = self.generate_grid(search_space)
        keys = list(grid.keys())
        combinations = list(itertools.product(*(grid[k] for k in keys)))
        
        best_score = -float('inf')
        best_params = None
        best_model = None
        
        print(f"Starte Grid Search mit {len(combinations)} Kombinationen...")
        
        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
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
                
            print(f"Kombination {idx+1}/{len(combinations)}: {params} -> Score (Fitness): {score:.6f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model_to_save
                
        return best_params, best_score, best_model
