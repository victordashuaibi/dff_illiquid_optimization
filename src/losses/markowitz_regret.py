"""Direct decision-regret loss for the penalty-form Markowitz objective.

This module replaces SPO+ for the quadratic-objective portfolio task. The
project's optimization problem is

    minimize   f(w, c, Σ) = -c'w + γ w'Σw
    s.t.       sum(w) = 1, w >= 0   (long-only simplex)

which is quadratic in ``w``. SPO+ (Elmachtoub & Grigas 2022) is derived
for *linear* objectives and loses its Fisher-consistency guarantee under
this objective. Instead we backpropagate the decision regret directly
through :class:`DiffMarkowitz` (cvxpylayers), the differentiable QP
solver. This is the Wilder 2019 / Donti-Amos-Kolter 2017 lineage with
DFF's bias-correction trust region wrapped on top.

We keep :mod:`src.losses.spo_plus` untouched — it remains useful for the
linear-objective Wilder 2019 LP replication.

Caching contract
----------------
``w*(c_true)`` does not change during training (ground truth is fixed),
so the trainer precomputes it once with :func:`build_oracle_cache` and
passes the resulting detached tensor to :class:`MarkowitzRegretLoss` at
init. Per-batch, the loss looks up oracle weights by integer index
(``w_oracle_cache[cache_idx]``) — no per-step QP solve for ``w*(c_true)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.optimizer.markowitz_diff import DiffMarkowitz
from src.optimizer.markowitz_static import MarkowitzStatic


def _markowitz_objective(
    c: torch.Tensor,
    w: torch.Tensor,
    Sigma: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute ``f(w, c, Σ) = -c'w + γ w'Σw`` per batch row.

    Shapes
    ------
    c, w  : ``(B, n_assets)``
    Sigma : ``(n_assets, n_assets)`` — shared across batch — OR
            ``(B, n_assets, n_assets)`` — per-instance.

    Returns ``(B,)``.
    """
    linear = -(c * w).sum(dim=-1)
    if Sigma.dim() == 2:
        quad = torch.einsum("bi,ij,bj->b", w, Sigma, w)
    elif Sigma.dim() == 3:
        if Sigma.shape[0] != w.shape[0]:
            raise ValueError(
                f"Sigma batch dim {Sigma.shape[0]} != w batch dim {w.shape[0]}"
            )
        quad = torch.einsum("bi,bij,bj->b", w, Sigma, w)
    else:
        raise ValueError(
            f"Sigma must be 2D (n,n) or 3D (B,n,n), got shape {tuple(Sigma.shape)}"
        )
    return linear + gamma * quad


def build_oracle_cache(
    c_true_array: np.ndarray,
    Sigma_array: np.ndarray,
    gamma: float,
    solver_static: MarkowitzStatic,
) -> torch.Tensor:
    """Solve the Markowitz oracle for every training instance once.

    Parameters
    ----------
    c_true_array
        Ground-truth returns for every training instance, shape
        ``(N, n_assets)``.
    Sigma_array
        Covariance matrices, shape ``(n_assets, n_assets)`` (shared) or
        ``(N, n_assets, n_assets)`` (per-instance).
    gamma
        Risk aversion. Must equal ``solver_static.risk_aversion``.
    solver_static
        Pre-configured :class:`MarkowitzStatic` whose ``solve_batch`` is
        the source of truth for ``w*``.

    Returns
    -------
    torch.Tensor
        Float64 CPU tensor of shape ``(N, n_assets)`` with
        ``requires_grad=False``. The trainer (Task 5) holds this and
        passes it to :class:`MarkowitzRegretLoss` at construction time.
    """
    if c_true_array.ndim != 2:
        raise ValueError(
            f"c_true_array must be 2D (N, n_assets), got shape {c_true_array.shape}"
        )
    if Sigma_array.ndim not in (2, 3):
        raise ValueError(
            f"Sigma_array must be 2D or 3D, got shape {Sigma_array.shape}"
        )
    if abs(float(gamma) - float(solver_static.risk_aversion)) > 1e-12:
        raise ValueError(
            f"gamma={gamma} != solver_static.risk_aversion="
            f"{solver_static.risk_aversion}; they must match exactly"
        )

    w_oracle_np = solver_static.solve_batch(c_true_array, Sigma_array)
    return torch.tensor(
        w_oracle_np, dtype=torch.float64, device="cpu", requires_grad=False
    ).detach()


class MarkowitzRegretLoss(nn.Module):
    """
    Direct decision-regret loss for penalty-form mean-variance:

        f(w, c) = -c'w + γ w'Σw
        regret(c, c̃) = f(w*(c̃), c) - f(w*(c), c)

    where ``w*(c̃)`` is computed via :class:`DiffMarkowitz` (cvxpylayers,
    differentiable in ``c̃``) and ``w*(c)`` is precomputed once per dataset
    and cached.

    This replaces SPO+ for our quadratic objective. Gradient flows
    through ``DiffMarkowitz`` → ``c̃`` → ``F_θ`` → ``θ`` via standard
    autograd. No SPO+ subgradient trick is needed because the objective
    is smooth in ``c̃``.

    Note on cache shape: ``w*(c)`` is computed once at trainer init time
    over the *training set* and stored as a tensor of shape
    ``(N_train, n_assets)``. Per-batch lookup is by integer index into
    this cache. The cache is detached and never participates in the
    autograd graph.

    Note on Sigma batching: :class:`DiffMarkowitz` accepts 2D ``Sigma``
    only (single shared covariance per call). When this loss receives a
    3D ``Sigma`` of shape ``(B, n, n)`` — the per-instance covariance
    case our portfolio loader produces — :meth:`forward` loops over the
    batch and calls ``DiffMarkowitz`` once per row. ``DiffMarkowitz``
    rebuilds its internal :class:`CvxpyLayer` whenever ``Sigma`` changes,
    so this path is **slow**. The trainer (Task 5) decides whether to
    pass shared 2D ``Sigma`` (fast) or per-instance 3D ``Sigma`` (slow
    but correct under heteroscedastic covariances).
    """

    def __init__(
        self,
        gamma: float,
        diff_markowitz: DiffMarkowitz,
        w_oracle_cache: torch.Tensor,
    ):
        """
        Parameters
        ----------
        gamma
            Risk aversion. Must match the ``gamma`` used to build
            ``w_oracle_cache`` and the one inside ``diff_markowitz``.
        diff_markowitz
            Pre-constructed differentiable optimizer.
        w_oracle_cache
            Detached float64 CPU tensor of shape
            ``(N_train, n_assets)`` produced by :func:`build_oracle_cache`.
            ``requires_grad`` must be False.
        """
        super().__init__()
        if w_oracle_cache.dim() != 2:
            raise ValueError(
                "w_oracle_cache must be 2D (N_train, n_assets), "
                f"got shape {tuple(w_oracle_cache.shape)}"
            )
        if w_oracle_cache.requires_grad:
            raise ValueError(
                "w_oracle_cache must be detached (requires_grad=False)"
            )
        if abs(float(gamma) - float(diff_markowitz.gamma)) > 1e-12:
            raise ValueError(
                f"gamma={gamma} != diff_markowitz.gamma={diff_markowitz.gamma}; "
                "they must match exactly"
            )

        self.gamma = float(gamma)
        self.diff_markowitz = diff_markowitz
        # Buffer (not Parameter): no gradient, but moves with the module on .to().
        self.register_buffer(
            "w_oracle_cache", w_oracle_cache.detach(), persistent=False
        )

    def forward(
        self,
        c_tilde: torch.Tensor,
        c_true: torch.Tensor,
        Sigma: torch.Tensor,
        cache_idx: torch.Tensor,
        validate: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        c_tilde
            ``(B, n_assets)`` corrected predictions from ``F_θ``. Carries
            the gradient signal back to the bias correction layer.
        c_true
            ``(B, n_assets)`` ground truth.
        Sigma
            ``(n_assets, n_assets)`` (shared) or
            ``(B, n_assets, n_assets)`` (per-instance).
        cache_idx
            ``(B,)`` int64 indices into ``w_oracle_cache``.
        validate
            If True, raise ``ValueError`` when any per-instance regret
            falls below ``-1e-4`` — sign-convention sentinel. Off by
            default to keep the training step lean.

        Returns
        -------
        torch.Tensor
            Scalar mean regret over the batch.
        """
        if c_tilde.shape != c_true.shape:
            raise ValueError(
                f"c_tilde shape {tuple(c_tilde.shape)} != c_true shape "
                f"{tuple(c_true.shape)}"
            )
        if c_tilde.dim() != 2:
            raise ValueError(
                f"c_tilde must be 2D (B, n_assets), got {tuple(c_tilde.shape)}"
            )
        B, n = c_tilde.shape
        if cache_idx.shape != (B,):
            raise ValueError(
                f"cache_idx shape {tuple(cache_idx.shape)} != (B={B},)"
            )
        if cache_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"cache_idx must be integer dtype, got {cache_idx.dtype}"
            )

        # Oracle weights: detached lookup, no grad path.
        w_true = self.w_oracle_cache[cache_idx]

        # w*(c̃) via the differentiable optimizer. DiffMarkowitz only takes
        # 2D Sigma — loop over the batch when Sigma is per-instance (3D).
        if Sigma.dim() == 2:
            w_pred = self.diff_markowitz(c_tilde, Sigma)
        elif Sigma.dim() == 3:
            if Sigma.shape[0] != B:
                raise ValueError(
                    f"Sigma batch dim {Sigma.shape[0]} != c_tilde batch dim {B}"
                )
            w_pred_rows = []
            for i in range(B):
                w_pred_rows.append(
                    self.diff_markowitz(c_tilde[i:i + 1], Sigma[i]).squeeze(0)
                )
            w_pred = torch.stack(w_pred_rows, dim=0)
        else:
            raise ValueError(
                f"Sigma must be 2D or 3D, got shape {tuple(Sigma.shape)}"
            )

        f_pred = _markowitz_objective(c_true, w_pred, Sigma, self.gamma)
        f_true = _markowitz_objective(c_true, w_true, Sigma, self.gamma)
        regret = f_pred - f_true

        if validate:
            min_regret = float(regret.detach().min().item())
            if min_regret < -1e-4:
                raise ValueError(
                    f"min per-instance Markowitz regret {min_regret:.3e} < -1e-4 "
                    "— sign convention bug, oracle is supposed to minimize"
                )

        return regret.mean()
