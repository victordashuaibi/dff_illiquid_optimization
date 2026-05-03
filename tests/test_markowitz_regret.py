"""Tests for the direct Markowitz decision-regret loss + oracle cache.

The contract pinned down here:

* ``regret(c_tilde=c_true) ≈ 0`` (within solver noise);
* random perturbations give per-sample regret ``>= -1e-4`` (sign convention);
* gradient flows from regret back to ``c_tilde`` (cvxpylayers backprop);
* gradient does NOT flow to the oracle cache (cache stays detached);
* the loss agrees with a two-way ``MarkowitzStatic``-only computation;
* :func:`build_oracle_cache` returns the right shape, dtype, and device,
  and matches per-instance ``MarkowitzStatic.solve`` calls.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.losses.markowitz_regret import (
    MarkowitzRegretLoss,
    _markowitz_objective,
    build_oracle_cache,
)
from src.optimizer.markowitz_diff import DiffMarkowitz
from src.optimizer.markowitz_static import MarkowitzStatic


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _make_psd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T
    Sigma = Sigma / np.max(np.abs(Sigma))
    return Sigma + 1e-3 * np.eye(n)


def _wire_loss(
    n_assets: int,
    n_train: int,
    gamma: float = 1.0,
    seed: int = 42,
) -> tuple[MarkowitzRegretLoss, MarkowitzStatic, np.ndarray, np.ndarray]:
    """Build a (loss, solver_static, c_true_np, Sigma_np) bundle.

    The cache covers all ``n_train`` instances; ``Sigma_np`` is shared
    across the batch (2D) for speed in tests where Sigma batching isn't
    the variable under test.
    """
    rng = np.random.default_rng(seed)
    c_true_np = rng.normal(size=(n_train, n_assets)) * 0.05
    Sigma_np = _make_psd(n_assets, seed=seed + 1)

    static = MarkowitzStatic(n_assets=n_assets, risk_aversion=gamma, long_only=True)
    cache = build_oracle_cache(c_true_np, Sigma_np, gamma=gamma, solver_static=static)
    diff = DiffMarkowitz(gamma=gamma)
    loss = MarkowitzRegretLoss(gamma=gamma, diff_markowitz=diff, w_oracle_cache=cache)
    return loss, static, c_true_np, Sigma_np


# ---------------------------------------------------------------------------
# 1. Regret ≈ 0 when c̃ = c_true
# ---------------------------------------------------------------------------
def test_regret_zero_when_c_tilde_equals_c_true():
    n_assets, B = 5, 4
    loss, _, c_true_np, Sigma_np = _wire_loss(n_assets=n_assets, n_train=B)

    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    c_tilde = c_true.clone()
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(B, dtype=torch.int64)

    out = loss(c_tilde, c_true, Sigma, cache_idx)
    assert torch.isfinite(out)
    # cvxpylayers SCS solver vs cvxpy CLARABEL/ECOS static solver: residuals
    # of ~1e-5 are normal; 1e-4 is a safe upper bound.
    assert abs(out.item()) < 1e-4, f"regret at c_tilde=c_true: {out.item():.6e}"


# ---------------------------------------------------------------------------
# 2. Random perturbations -> per-sample regret >= -1e-4
# ---------------------------------------------------------------------------
def test_regret_nonnegative_for_random_perturbations():
    n_assets, B = 5, 50
    loss, _, c_true_np, Sigma_np = _wire_loss(n_assets=n_assets, n_train=B, seed=7)

    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(B, dtype=torch.int64)

    rng = np.random.default_rng(123)
    c_tilde_np = c_true_np + 0.1 * rng.standard_normal(c_true_np.shape)
    c_tilde = torch.tensor(c_tilde_np, dtype=torch.float64)

    # Use the per-row regret directly, not the mean, so we catch a single
    # negative outlier.
    w_pred = loss.diff_markowitz(c_tilde, Sigma)
    w_true = loss.w_oracle_cache  # the cache spans the batch 1:1
    f_pred = _markowitz_objective(c_true, w_pred, Sigma, loss.gamma)
    f_true = _markowitz_objective(c_true, w_true, Sigma, loss.gamma)
    per_row = (f_pred - f_true).detach().cpu().numpy()

    min_regret = float(per_row.min())
    assert min_regret > -1e-4, (
        f"sign convention bug: min per-row regret = {min_regret:.3e} "
        f"(< -1e-4); expected oracle to be a true minimizer"
    )
    # Also confirm there's actual signal — perturbed predictions should
    # produce strictly positive regret somewhere.
    assert per_row.max() > 1e-4, (
        f"max per-row regret {per_row.max():.3e} suggests perturbations "
        "don't move the optimum at all — fixture may be degenerate"
    )


# ---------------------------------------------------------------------------
# 3. Gradient flows to c_tilde
# ---------------------------------------------------------------------------
def test_gradient_flows_to_c_tilde():
    n_assets, B = 5, 4
    loss, _, c_true_np, Sigma_np = _wire_loss(n_assets=n_assets, n_train=B, seed=3)

    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(B, dtype=torch.int64)

    init = c_true_np + 0.05 * np.random.default_rng(9).standard_normal(c_true_np.shape)
    c_tilde = torch.nn.Parameter(torch.tensor(init, dtype=torch.float64))

    out = loss(c_tilde, c_true, Sigma, cache_idx)
    out.backward()

    assert c_tilde.grad is not None, "c_tilde.grad is None — backprop missing"
    assert torch.isfinite(c_tilde.grad).all(), "c_tilde.grad has non-finite entries"
    assert (c_tilde.grad.abs() > 0).any(), (
        "c_tilde.grad is all zeros — gradient signal is dead"
    )


# ---------------------------------------------------------------------------
# 4. Gradient does NOT flow to the oracle cache
# ---------------------------------------------------------------------------
def test_gradient_does_not_flow_to_cache():
    n_assets, B = 4, 3
    loss, _, c_true_np, Sigma_np = _wire_loss(n_assets=n_assets, n_train=B, seed=5)

    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(B, dtype=torch.int64)

    # Cache must declare requires_grad=False at construction.
    assert loss.w_oracle_cache.requires_grad is False
    cache_id_before = id(loss.w_oracle_cache)

    c_tilde = torch.nn.Parameter(c_true.clone() + 0.05)
    out = loss(c_tilde, c_true, Sigma, cache_idx)
    out.backward()

    # Cache identity unchanged, still no grad, no .grad attribute populated.
    assert id(loss.w_oracle_cache) == cache_id_before
    assert loss.w_oracle_cache.requires_grad is False
    assert loss.w_oracle_cache.grad is None, (
        "w_oracle_cache.grad got populated — someone removed the .detach()"
    )


# ---------------------------------------------------------------------------
# 5a. Analytical 1-asset case (regret must be 0 by feasibility)
# ---------------------------------------------------------------------------
def test_analytical_regret_in_one_d_is_zero():
    """n_assets=1 + simplex (sum=1, w>=0) has only one feasible point: w=[1].

    So ``w*(c̃) = w*(c) = [1]`` regardless of ``c̃``, and the regret is
    identically zero (up to solver noise).
    """
    gamma = 0.5
    c_true_np = np.array([[1.0], [-0.5]])  # B=2, n_assets=1
    Sigma_np = np.array([[1.0]])

    static = MarkowitzStatic(n_assets=1, risk_aversion=gamma, long_only=True)
    cache = build_oracle_cache(c_true_np, Sigma_np, gamma=gamma, solver_static=static)
    diff = DiffMarkowitz(gamma=gamma)
    loss = MarkowitzRegretLoss(gamma=gamma, diff_markowitz=diff, w_oracle_cache=cache)

    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    c_tilde = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)  # arbitrary
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(2, dtype=torch.int64)

    out = loss(c_tilde, c_true, Sigma, cache_idx)
    assert abs(out.item()) < 1e-4


# ---------------------------------------------------------------------------
# 5b. Analytical 2-asset case (interior optima, closed-form regret)
# ---------------------------------------------------------------------------
def test_analytical_regret_in_two_d_matches_closed_form():
    """For n_assets=2 with diagonal ``Sigma = σ²·I`` and gamma=1, the
    long-only simplex Markowitz QP has interior optimum

        a* = 0.5 + (c0 - c1) / (4 γ σ²),     w* = (a*, 1-a*)

    when ``a* ∈ [0, 1]``. With ``c=(0.05, 0.02)``, ``c̃=(0.06, 0.01)``,
    ``γ=1``, ``σ²=0.04``, both ``a_true=0.6875`` and ``a_pred=0.8125``
    are interior, and the analytical regret works out to 0.00125.
    """
    gamma = 1.0
    sigma_sq = 0.04
    c_true_np = np.array([[0.05, 0.02]])
    c_tilde_np = np.array([[0.06, 0.01]])
    Sigma_np = sigma_sq * np.eye(2)

    static = MarkowitzStatic(n_assets=2, risk_aversion=gamma, long_only=True)
    cache = build_oracle_cache(c_true_np, Sigma_np, gamma=gamma, solver_static=static)
    diff = DiffMarkowitz(gamma=gamma)
    loss = MarkowitzRegretLoss(gamma=gamma, diff_markowitz=diff, w_oracle_cache=cache)

    out = loss(
        torch.tensor(c_tilde_np, dtype=torch.float64),
        torch.tensor(c_true_np, dtype=torch.float64),
        torch.tensor(Sigma_np, dtype=torch.float64),
        torch.tensor([0], dtype=torch.int64),
    )
    expected_regret = 0.00125  # see test docstring
    assert abs(out.item() - expected_regret) < 1e-4, (
        f"closed-form regret mismatch: got {out.item():.6e}, expected {expected_regret}"
    )


# ---------------------------------------------------------------------------
# 5c. Two-way agreement: loss vs. MarkowitzStatic-only computation (fallback)
# ---------------------------------------------------------------------------
def test_regret_matches_markowitz_static_two_way():
    """Compute regret via two paths and assert they agree:

    1. The loss class (DiffMarkowitz for ``w_pred``, cached ``w_true``).
    2. Manual: solve both ``w*(c̃)`` and ``w*(c)`` with MarkowitzStatic
       (the static solver already used for the cache), evaluate the
       quadratic objective, take the difference.

    A 1e-3 gap is acceptable — DiffMarkowitz uses SCS; MarkowitzStatic
    uses cvxpy's default cone solver. The point is that we're not off
    by an order of magnitude or a sign.
    """
    n_assets, B = 6, 8
    gamma = 1.0
    loss, static, c_true_np, Sigma_np = _wire_loss(
        n_assets=n_assets, n_train=B, gamma=gamma, seed=11
    )

    rng = np.random.default_rng(31)
    c_tilde_np = c_true_np + 0.1 * rng.standard_normal(c_true_np.shape)

    # Path 1: the loss class.
    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    c_tilde = torch.tensor(c_tilde_np, dtype=torch.float64)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(B, dtype=torch.int64)
    regret_loss = loss(c_tilde, c_true, Sigma, cache_idx).item()

    # Path 2: MarkowitzStatic for both w*(c̃) and w*(c), manual objective.
    w_pred_static = static.solve_batch(c_tilde_np, Sigma_np)
    w_true_static = static.solve_batch(c_true_np, Sigma_np)

    def f(w, c):
        # f(w, c) = -c'w + γ w'Σw  (same convention as the loss).
        linear = -(c * w).sum(axis=-1)
        quad = np.einsum("bi,ij,bj->b", w, Sigma_np, w)
        return linear + gamma * quad

    f_pred = f(w_pred_static, c_true_np)
    f_true = f(w_true_static, c_true_np)
    regret_static = float((f_pred - f_true).mean())

    assert abs(regret_loss - regret_static) < 1e-3, (
        f"loss-class regret {regret_loss:.6e} != static two-way "
        f"{regret_static:.6e} (|Δ|={abs(regret_loss - regret_static):.3e})"
    )


# ---------------------------------------------------------------------------
# 6. build_oracle_cache: shapes / dtypes / device / requires_grad
# ---------------------------------------------------------------------------
def test_build_oracle_cache_shapes_and_dtypes():
    rng = np.random.default_rng(0)
    N, n = 7, 5
    gamma = 0.5
    c_true_np = rng.normal(size=(N, n)) * 0.05
    Sigma_np = np.stack([_make_psd(n, seed=i) for i in range(N)], axis=0)
    assert Sigma_np.shape == (N, n, n)

    static = MarkowitzStatic(n_assets=n, risk_aversion=gamma, long_only=True)
    cache = build_oracle_cache(c_true_np, Sigma_np, gamma=gamma, solver_static=static)

    assert isinstance(cache, torch.Tensor)
    assert cache.shape == (N, n)
    assert cache.dtype == torch.float64
    assert cache.requires_grad is False
    assert cache.device.type == "cpu"


# ---------------------------------------------------------------------------
# 7. build_oracle_cache: matches per-instance MarkowitzStatic.solve
# ---------------------------------------------------------------------------
def test_build_oracle_cache_matches_per_instance_solve():
    rng = np.random.default_rng(99)
    N, n = 6, 4
    gamma = 1.0
    c_true_np = rng.normal(size=(N, n)) * 0.05
    Sigma_np = np.stack([_make_psd(n, seed=i + 10) for i in range(N)], axis=0)

    static = MarkowitzStatic(n_assets=n, risk_aversion=gamma, long_only=True)
    cache = build_oracle_cache(c_true_np, Sigma_np, gamma=gamma, solver_static=static)
    cache_np = cache.cpu().numpy()

    for i in range(N):
        w_i = static.solve(c_true_np[i], Sigma_np[i])
        assert np.max(np.abs(cache_np[i] - w_i)) < 1e-6, (
            f"row {i}: cache vs solve disagree by {np.max(np.abs(cache_np[i] - w_i)):.3e}"
        )


# ---------------------------------------------------------------------------
# Extra: error handling for misuse
# ---------------------------------------------------------------------------
def test_loss_rejects_non_detached_cache():
    n = 4
    cache = torch.randn(3, n, dtype=torch.float64, requires_grad=True)
    diff = DiffMarkowitz(gamma=1.0)
    with pytest.raises(ValueError, match="detached"):
        MarkowitzRegretLoss(gamma=1.0, diff_markowitz=diff, w_oracle_cache=cache)


def test_loss_rejects_gamma_mismatch_with_diff_markowitz():
    cache = torch.zeros(2, 3, dtype=torch.float64, requires_grad=False)
    diff = DiffMarkowitz(gamma=2.0)
    with pytest.raises(ValueError, match="gamma"):
        MarkowitzRegretLoss(gamma=1.0, diff_markowitz=diff, w_oracle_cache=cache)


def test_build_oracle_cache_rejects_gamma_mismatch():
    static = MarkowitzStatic(n_assets=3, risk_aversion=2.0)
    with pytest.raises(ValueError, match="risk_aversion"):
        build_oracle_cache(
            np.zeros((1, 3)), np.eye(3), gamma=1.0, solver_static=static
        )


def test_validate_flag_catches_bogus_cache():
    """Inject a bogus oracle (worse than the prediction) and confirm
    ``validate=True`` raises. This is the sign-convention sentinel.
    """
    n_assets, B = 4, 3
    gamma = 1.0
    rng = np.random.default_rng(2)
    c_true_np = rng.normal(size=(B, n_assets)) * 0.05
    Sigma_np = _make_psd(n_assets, seed=2)

    static = MarkowitzStatic(n_assets=n_assets, risk_aversion=gamma, long_only=True)
    real_cache = build_oracle_cache(c_true_np, Sigma_np, gamma=gamma, solver_static=static)
    # Replace the oracle with the uniform portfolio — guaranteed worse than
    # the true optimum on most problems, so f(uniform) > f(w*(c̃)) often.
    bogus_cache = torch.full_like(real_cache, fill_value=1.0 / n_assets)

    diff = DiffMarkowitz(gamma=gamma)
    loss = MarkowitzRegretLoss(gamma=gamma, diff_markowitz=diff, w_oracle_cache=bogus_cache)

    c_true = torch.tensor(c_true_np, dtype=torch.float64)
    c_tilde = c_true.clone()
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)
    cache_idx = torch.arange(B, dtype=torch.int64)

    # validate=False: silently returns whatever value (possibly negative).
    out_silent = loss(c_tilde, c_true, Sigma, cache_idx, validate=False)
    assert torch.isfinite(out_silent)

    # validate=True: must raise because at least one row's regret is < -1e-4
    # (uniform portfolio is not the optimum at c̃ = c_true).
    with pytest.raises(ValueError, match="sign convention"):
        loss(c_tilde, c_true, Sigma, cache_idx, validate=True)
