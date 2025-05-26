from turbo.turbo_m import TurboM
import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import os

class Turbo:
    def __init__(self, function, dim, function_name, n_init, max_evals, seed=0):
        self.function = function
        self.dim = dim
        self.function_name = function_name
        self.max_evals = max_evals
        self.seed = seed
        self.n_init = n_init

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

    def run(self):
        def f(x):
            x = torch.tensor(x, dtype=torch.float64)
            return self.function(x).item()

        turbo = TurboM(
            f=f,
            lb=self.lb,
            ub=self.ub,
            n_init=self.n_init,
            max_evals=self.max_evals,
            n_trust_regions=5,
            batch_size=10,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            dtype="float64",
        )

        turbo.optimize()

        self.X = turbo.X  # Evaluated points
        self.fX = turbo.fX  # Observed values
        ind_best = np.argmin(self.fX)
        self.f_best, self.x_best = self.fX[ind_best], self.X[ind_best, :]

        self._save_results()

    def _save_results(self):
        save_dir = f"results_turbo/{self.function_name}_{self.dim}d"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/X.npy", self.X)
        np.save(f"{save_dir}/fX.npy", self.fX)