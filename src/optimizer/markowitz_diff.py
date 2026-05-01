import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class DiffMarkowitz(nn.Module):
    def __init__(self, gamma: float = 1.0, solver_args: dict | None = None):
        super().__init__()
        self.gamma = gamma
        self.solver_args = solver_args or {"eps": 1e-8, "max_iters": 100000}

        self.n_assets = None
        self.layer = None
        self.sigma_cache = None

    def _make_layer(self, Sigma: torch.Tensor):
        Sigma_np = Sigma.detach().cpu().numpy()
        Sigma_np = 0.5 * (Sigma_np + Sigma_np.T)
        Sigma_np = Sigma_np + 1e-8 * np.eye(Sigma_np.shape[0])

        n_assets = Sigma_np.shape[0]

        w = cp.Variable(n_assets)
        c = cp.Parameter(n_assets)

        objective = cp.Minimize(
            -c @ w + self.gamma * cp.quad_form(w, Sigma_np)
        )

        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        problem = cp.Problem(objective, constraints)

        if not problem.is_dpp():
            raise ValueError("The CVXPY problem is not DPP.")

        self.n_assets = n_assets
        self.sigma_cache = Sigma_np
        self.layer = CvxpyLayer(
            problem,
            parameters=[c],
            variables=[w]
        )

    def forward(self, c_hat: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        if c_hat.dim() != 2:
            raise ValueError("c_hat must have shape [batch, n_assets].")

        if Sigma.dim() != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must have shape [n_assets, n_assets].")

        batch_size, n_assets = c_hat.shape

        if Sigma.shape[0] != n_assets:
            raise ValueError("Sigma dimension must match c_hat.shape[1].")

        rebuild = False

        if self.layer is None:
            rebuild = True
        elif self.n_assets != n_assets:
            rebuild = True
        elif self.sigma_cache is None:
            rebuild = True
        else:
            Sigma_np = Sigma.detach().cpu().numpy()
            Sigma_np = 0.5 * (Sigma_np + Sigma_np.T)

            if not np.allclose(Sigma_np, self.sigma_cache, atol=1e-10):
                rebuild = True

        if rebuild:
            self._make_layer(Sigma)

        weights, = self.layer(c_hat, solver_args=self.solver_args)

        return weights