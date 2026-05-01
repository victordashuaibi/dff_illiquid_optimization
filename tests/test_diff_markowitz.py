import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import cvxpy as cp

from src.optimizer.markowitz_diff import DiffMarkowitz


def make_psd_matrix(n_assets: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_assets, n_assets))
    Sigma = A.T @ A
    Sigma = Sigma / np.max(np.abs(Sigma))
    Sigma = Sigma + 1e-3 * np.eye(n_assets)
    return Sigma


def solve_cvxpy_markowitz(c, Sigma, gamma=1.0):
    n_assets = len(c)

    w = cp.Variable(n_assets)

    objective = cp.Minimize(
        -c @ w + gamma * cp.quad_form(w, Sigma)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if w.value is None:
        problem.solve(solver=cp.SCS, eps=1e-8, verbose=False)

    return w.value


def test_diff_markowitz_weights_sum_to_one():
    torch.manual_seed(42)

    n_assets = 10
    batch_size = 4

    c_hat = torch.randn(batch_size, n_assets, dtype=torch.float64)
    Sigma_np = make_psd_matrix(n_assets)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    optimizer = DiffMarkowitz(gamma=1.0)
    w = optimizer(c_hat, Sigma)

    assert w.shape == (batch_size, n_assets)
    assert torch.allclose(
        w.sum(dim=1),
        torch.ones(batch_size, dtype=w.dtype),
        atol=1e-5
    )


def test_diff_markowitz_weights_nonnegative():
    torch.manual_seed(42)

    n_assets = 10
    batch_size = 4

    c_hat = torch.randn(batch_size, n_assets, dtype=torch.float64)
    Sigma_np = make_psd_matrix(n_assets)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    optimizer = DiffMarkowitz(gamma=1.0)
    w = optimizer(c_hat, Sigma)

    assert torch.min(w).item() >= -1e-6


def test_diff_markowitz_is_differentiable_wrt_c_hat():
    torch.manual_seed(42)

    n_assets = 10
    batch_size = 3

    c_hat = torch.randn(
        batch_size,
        n_assets,
        dtype=torch.float64,
        requires_grad=True
    )

    Sigma_np = make_psd_matrix(n_assets)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    optimizer = DiffMarkowitz(gamma=1.0)
    w = optimizer(c_hat, Sigma)

    loss = (w * c_hat).sum()
    grad, = torch.autograd.grad(loss, c_hat, retain_graph=True)

    assert grad is not None
    assert grad.shape == c_hat.shape
    assert torch.isfinite(grad).all()


def test_diff_markowitz_matches_cvxpy_solution():
    torch.manual_seed(42)
    np.random.seed(42)

    n_assets = 10
    batch_size = 3

    c_hat_np = np.random.normal(size=(batch_size, n_assets))
    Sigma_np = make_psd_matrix(n_assets)

    c_hat = torch.tensor(c_hat_np, dtype=torch.float64)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    optimizer = DiffMarkowitz(gamma=1.0)
    w_diff = optimizer(c_hat, Sigma).detach().cpu().numpy()

    for i in range(batch_size):
        w_cvx = solve_cvxpy_markowitz(
            c=c_hat_np[i],
            Sigma=Sigma_np,
            gamma=1.0
        )

        assert np.max(np.abs(w_diff[i] - w_cvx)) < 1e-4