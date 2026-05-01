"""Decision regret and Normalized Decision Regret (NDR).

Two flavours are provided, one per type of downstream optimization
objective; see ``docs/INTERFACE.md`` for the contract:

* :func:`decision_regret` / :func:`normalized_decision_regret` — for
  *linear* minimization objectives ``f(w, c) = c^T w`` (used by SPO/SPO+
  on shortest-path, matching, knapsack, etc.).
* :func:`markowitz_decision_regret` / :func:`markowitz_normalized_decision_regret`
  — for *quadratic* Markowitz mean-variance objectives
  ``f(w, c, Σ) = -c^T w + γ w^T Σ w`` (used by every portfolio
  experiment in this project).
"""
import torch


# ---------------------------------------------------------------------------
# Linear-objective regret (kept for SPO+ / shortest-path style tasks)
# ---------------------------------------------------------------------------
def decision_regret(c_true: torch.Tensor, w_pred: torch.Tensor,
                    w_oracle: torch.Tensor) -> torch.Tensor:
    """
    DR(c, c_hat) = f(w*(c_hat), c) - f(w*(c), c)
    For linear objective f(w, c) = c^T w (minimization).

    Note
    ----
    Assumes the *linear* minimization objective f(w, c) = c^T w. For
    Markowitz mean-variance problems, use :func:`markowitz_decision_regret`
    or :func:`markowitz_normalized_decision_regret` instead.
    """
    return (c_true * w_pred).sum(dim=-1) - (c_true * w_oracle).sum(dim=-1)


def normalized_decision_regret(c_true: torch.Tensor, w_pred: torch.Tensor,
                               w_oracle: torch.Tensor) -> torch.Tensor:
    """NDR per Tang & Khalil 2022 / DFF paper Eq. 19, for linear objectives.

    Note
    ----
    Assumes the *linear* minimization objective f(w, c) = c^T w. For
    Markowitz mean-variance problems, use
    :func:`markowitz_normalized_decision_regret` instead.
    """
    obj_pred = (c_true * w_pred).sum(dim=-1)
    obj_oracle = (c_true * w_oracle).sum(dim=-1)
    return (obj_pred - obj_oracle).sum() / obj_oracle.abs().sum().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Markowitz quadratic-objective regret (used by every portfolio experiment)
# ---------------------------------------------------------------------------
def _markowitz_objective(
    c: torch.Tensor,
    w: torch.Tensor,
    Sigma: torch.Tensor,
    risk_aversion: float,
) -> torch.Tensor:
    """Compute ``f(w, c, Σ) = -c^T w + γ w^T Σ w`` per instance.

    Shapes:
      - c, w : ``[batch, n_assets]``
      - Sigma: ``[n_assets, n_assets]`` (shared) or ``[batch, n_assets, n_assets]``
    Returns ``[batch]``.
    """
    linear = -(c * w).sum(dim=-1)
    if Sigma.dim() == 2:
        # quadratic = w_b^T Σ w_b for each row b
        quad = torch.einsum("bi,ij,bj->b", w, Sigma, w)
    elif Sigma.dim() == 3:
        if Sigma.shape[0] != w.shape[0]:
            raise ValueError(
                f"Sigma batch dim {Sigma.shape[0]} != w batch dim {w.shape[0]}"
            )
        quad = torch.einsum("bi,bij,bj->b", w, Sigma, w)
    else:
        raise ValueError(
            f"Sigma must be 2D [n,n] or 3D [B,n,n], got shape {tuple(Sigma.shape)}"
        )
    return linear + risk_aversion * quad


def markowitz_decision_regret(
    c_true: torch.Tensor,
    w_pred: torch.Tensor,
    w_oracle: torch.Tensor,
    Sigma: torch.Tensor,
    risk_aversion: float = 1.0,
) -> torch.Tensor:
    """Per-instance Markowitz decision regret using the full quadratic objective.

    ``f(w, c, Σ) = -c^T w + γ w^T Σ w`` (minimization form, matching
    :class:`src.optimizer.markowitz_static.MarkowitzStatic`).

    Returns
    -------
    torch.Tensor
        Shape ``[batch]``; entries are guaranteed ``>= 0`` (up to solver
        tolerance) since ``w_oracle`` minimizes ``f`` at ``c_true``.
    """
    f_pred = _markowitz_objective(c_true, w_pred, Sigma, risk_aversion)
    f_oracle = _markowitz_objective(c_true, w_oracle, Sigma, risk_aversion)
    return f_pred - f_oracle


def markowitz_normalized_decision_regret(
    c_true: torch.Tensor,
    w_pred: torch.Tensor,
    w_oracle: torch.Tensor,
    Sigma: torch.Tensor,
    risk_aversion: float = 1.0,
) -> torch.Tensor:
    """Markowitz NDR — same shape as DFF Eq. 19 but with the quadratic objective.

    ``NDR = sum_b (f(w_pred_b) - f(w_oracle_b)) / sum_b |f(w_oracle_b)|``

    where ``f(w, c, Σ) = -c^T w + γ w^T Σ w``.

    Returns a scalar tensor.
    """
    f_pred = _markowitz_objective(c_true, w_pred, Sigma, risk_aversion)
    f_oracle = _markowitz_objective(c_true, w_oracle, Sigma, risk_aversion)
    numerator = (f_pred - f_oracle).sum()
    denom = f_oracle.abs().sum().clamp(min=1e-8)
    return numerator / denom
