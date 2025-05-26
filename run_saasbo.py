import os
import torch
import numpy as np
import sys
sys.path.append("/Users/teresa/Desktop/bo_search_space/saasbo_repo")

from saasbo import run_saasbo


class SAASBO:
    def __init__(self, function, dim, function_name, n_init, max_evals, seed=0):
        self.function = function
        self.dim = dim
        self.function_name = function_name
        self.seed = seed
        self.n_init = n_init
        self.max_evals = max_evals
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def run(self):
        def f(x):
            x = torch.tensor(x, dtype=torch.float64)
            return self.function(x).item()

        # You must modify run_saasbo to return X, fX!
        X, fX = run_saasbo(
            f=f,
            lb=self.lb,
            ub=self.ub,
            max_evals=self.max_evals,
            num_init_evals=self.n_init,
            seed=self.seed,
            alpha=0.01,
            num_warmup=64,
            num_samples=64,
            thinning=8,
        )

        # Save
        save_dir = f"results_saasbo/{self.function_name}_{self.dim}d"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/X.npy", X)
        np.save(f"{save_dir}/fX.npy", fX)
