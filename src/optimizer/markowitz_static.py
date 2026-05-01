"""Static (non-differentiable) Markowitz solver via cvxpy.

Solves the contract from ``docs/INTERFACE.md``::

    minimize  -c^T w + gamma * w^T Sigma w
    s.t.      sum(w) = 1
              w >= 0          (if long_only=True)

Used as the baseline optimizer (paired with the XGBoostBackbone) and as
the oracle solver for computing decision regret.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


class MarkowitzStatic:
    def __init__(
        self,
        n_assets: int,
        risk_aversion: float = 1.0,
        long_only: bool = True,
    ):
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        self.n_assets = int(n_assets)
        self.risk_aversion = float(risk_aversion)
        self.long_only = bool(long_only)

        # Build a single parametric problem and reuse it across calls; cvxpy
        # caches the canonicalisation, so per-solve overhead drops sharply.
        self._w = cp.Variable(self.n_assets)
        self._c_param = cp.Parameter(self.n_assets)
        # Use a Cholesky-shaped parameter so the quadratic form stays DCP.
        self._L_param = cp.Parameter((self.n_assets, self.n_assets))

        objective = cp.Minimize(
            -self._c_param @ self._w
            + self.risk_aversion * cp.sum_squares(self._L_param @ self._w)
        )
        constraints = [cp.sum(self._w) == 1.0]
        if self.long_only:
            constraints.append(self._w >= 0)
        self._problem = cp.Problem(objective, constraints)

    @staticmethod
    def _cholesky_psd(Sigma: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
        """Return ``L`` with ``L L^T = Sigma``, adding jitter if needed."""
        S = 0.5 * (Sigma + Sigma.T)
        eps = jitter
        for _ in range(8):
            try:
                return np.linalg.cholesky(S + eps * np.eye(S.shape[0]))
            except np.linalg.LinAlgError:
                eps *= 10.0
        # Fallback: eigen-decomposition with non-negative clipping.
        vals, vecs = np.linalg.eigh(S)
        vals = np.clip(vals, a_min=0.0, a_max=None)
        return vecs * np.sqrt(vals)

    def solve(self, c: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float).reshape(-1)
        Sigma = np.asarray(Sigma, dtype=float)
        if c.shape != (self.n_assets,):
            raise ValueError(f"c shape {c.shape} != ({self.n_assets},)")
        if Sigma.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Sigma shape {Sigma.shape} != ({self.n_assets}, {self.n_assets})"
            )

        self._c_param.value = c
        self._L_param.value = self._cholesky_psd(Sigma).T  # so L^T w == sqrt(quad form)
        self._problem.solve(warm_start=True)

        if self._problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Markowitz solve failed: status={self._problem.status}")

        w = np.asarray(self._w.value, dtype=float).reshape(-1)
        if self.long_only:
            w = np.clip(w, a_min=0.0, a_max=None)
        s = w.sum()
        if s > 0:
            w = w / s
        return w

    def solve_batch(self, c: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Solve a batch sequentially. ``Sigma`` may be ``[n,n]`` or ``[B,n,n]``."""
        c = np.asarray(c, dtype=float)
        Sigma = np.asarray(Sigma, dtype=float)
        if c.ndim != 2:
            raise ValueError(f"c must be 2D [B, n_assets], got shape {c.shape}")
        B = c.shape[0]

        if Sigma.ndim == 2:
            Sigmas = [Sigma] * B
        elif Sigma.ndim == 3 and Sigma.shape[0] == B:
            Sigmas = list(Sigma)
        else:
            raise ValueError(
                f"Sigma must be [n,n] or [B,n,n] with B={B}, got shape {Sigma.shape}"
            )

        out = np.empty_like(c)
        for i in range(B):
            out[i] = self.solve(c[i], Sigmas[i])
        return out
