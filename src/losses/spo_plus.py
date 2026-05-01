from __future__ import annotations

import torch
import torch.nn as nn

from src.optimizer.markowitz_diff import DiffMarkowitz


class SPOPlusLoss(nn.Module):
    def __init__(self, gamma: float = 1.0, solver_args: dict | None = None):
        super().__init__()
        self.gamma = gamma
        self.optimizer = DiffMarkowitz(
            gamma=gamma,
            solver_args=solver_args
        )

    def portfolio_objective(
        self,
        returns: torch.Tensor,
        weights: torch.Tensor,
        Sigma: torch.Tensor
    ) -> torch.Tensor:
        linear_term = -torch.sum(returns * weights, dim=1)

        risk_term = torch.einsum(
            "bi,ij,bj->b",
            weights,
            Sigma,
            weights
        )

        objective_value = linear_term + self.gamma * risk_term

        return objective_value

    def forward(
        self,
        c_true: torch.Tensor,
        c_tilde: torch.Tensor,
        Sigma: torch.Tensor
    ) -> torch.Tensor:
        if c_true.dim() != 2:
            raise ValueError("c_true must have shape [batch, n_assets].")

        if c_tilde.dim() != 2:
            raise ValueError("c_tilde must have shape [batch, n_assets].")

        if c_true.shape != c_tilde.shape:
            raise ValueError("c_true and c_tilde must have the same shape.")

        if Sigma.dim() != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must have shape [n_assets, n_assets].")

        if Sigma.shape[0] != c_true.shape[1]:
            raise ValueError("Sigma dimension must match number of assets.")

        spo_signal = 2.0 * c_tilde - c_true

        w_spo = self.optimizer(spo_signal, Sigma)

        with torch.no_grad():
            w_true = self.optimizer(c_true, Sigma)

        obj_spo = self.portfolio_objective(
            returns=c_true,
            weights=w_spo,
            Sigma=Sigma
        )

        obj_true = self.portfolio_objective(
            returns=c_true,
            weights=w_true,
            Sigma=Sigma
        )

        loss = obj_spo - obj_true

        return loss.mean()