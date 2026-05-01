"""Decision regret and Normalized Decision Regret (NDR) — Eq. 19 of DFF paper."""
import torch


def decision_regret(c_true: torch.Tensor, w_pred: torch.Tensor,
                    w_oracle: torch.Tensor) -> torch.Tensor:
    """
    DR(c, c_hat) = f(w*(c_hat), c) - f(w*(c), c)
    For linear objective f(w, c) = c^T w (minimization).
    """
    return (c_true * w_pred).sum(dim=-1) - (c_true * w_oracle).sum(dim=-1)


def normalized_decision_regret(c_true: torch.Tensor, w_pred: torch.Tensor,
                               w_oracle: torch.Tensor) -> torch.Tensor:
    """NDR per Tang & Khalil 2022, used as primary metric in DFF paper."""
    obj_pred = (c_true * w_pred).sum(dim=-1)
    obj_oracle = (c_true * w_oracle).sum(dim=-1)
    return (obj_pred - obj_oracle).sum() / obj_oracle.abs().sum().clamp(min=1e-8)
