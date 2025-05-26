import torch
import numpy as np
import botorch
from run_turbo import Turbo
from run_saasbo import SAASBO

def ackley(X):
    """
    X in [0, 1]^d, scaled to [-32.768, 32.768]^d
    """
    X = 64.536 * (X - 0.5)  # scale to [-32.768, 32.768]
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = X.shape[-1]

    sum1 = torch.sum(X ** 2, dim=-1)
    sum2 = torch.sum(torch.cos(c * X), dim=-1)
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / d))
    term2 = -torch.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

def rastrigin(X):
    """
    X in [0, 1]^d, scaled to [-5.12, 5.12]^d
    """
    X = 10.24 * (X - 0.5)
    A = 10
    d = X.shape[-1]
    return A * d + torch.sum(X**2 - A  * torch.cos(2 * np.pi * X), dim=-1)


if __name__ == "__main__":
    """
    dims = [5, 10, 20, 30, 50, 75, 100, 125, 150]
    for dim in dims:
        Turbo(ackley, dim=dim, function_name="ackley", n_init=20, max_evals=400).run()
        Turbo(rastrigin, dim=dim, function_name="rastrigin", n_init=20, max_evals=400).run()
        SAASBO(ackley, dim=dim, function_name="ackley", n_init=max(dim, 20), max_evals=400).run()
        SAASBO(rastrigin, dim=dim, function_name="rastrigin", n_init=max(dim, 20), max_evals=400).run()
    """

    Turbo(rastrigin, dim=100, function_name="rastrigin", n_init=20, max_evals=400).run()