import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.losses.spo_plus import SPOPlusLoss


def make_psd_matrix(n_assets: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_assets, n_assets))
    Sigma = A.T @ A
    Sigma = Sigma / np.max(np.abs(Sigma))
    Sigma = Sigma + 1e-3 * np.eye(n_assets)
    return Sigma


def test_spo_plus_loss_zero_when_prediction_is_true():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 3
    n_assets = 10

    c_true = torch.randn(batch_size, n_assets, dtype=torch.float64)
    c_tilde = c_true.clone().detach().requires_grad_(True)

    Sigma_np = make_psd_matrix(n_assets)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    loss_fn = SPOPlusLoss(gamma=1.0)
    loss = loss_fn(c_true, c_tilde, Sigma)

    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-4


def test_spo_plus_loss_is_differentiable_wrt_c_tilde():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 3
    n_assets = 10

    c_true = torch.randn(batch_size, n_assets, dtype=torch.float64)
    c_tilde = (
        c_true + 0.05 * torch.randn(batch_size, n_assets, dtype=torch.float64)
    ).requires_grad_(True)

    Sigma_np = make_psd_matrix(n_assets)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    loss_fn = SPOPlusLoss(gamma=1.0)
    loss = loss_fn(c_true, c_tilde, Sigma)

    loss.backward()

    assert torch.isfinite(loss)
    assert c_tilde.grad is not None
    assert c_tilde.grad.shape == c_tilde.shape
    assert torch.isfinite(c_tilde.grad).all()


def test_spo_plus_loss_can_update_small_network():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 4
    n_assets = 10
    input_dim = 6

    X = torch.randn(batch_size, input_dim, dtype=torch.float64)
    c_true = torch.randn(batch_size, n_assets, dtype=torch.float64)

    model = nn.Linear(input_dim, n_assets).double()

    Sigma_np = make_psd_matrix(n_assets)
    Sigma = torch.tensor(Sigma_np, dtype=torch.float64)

    loss_fn = SPOPlusLoss(gamma=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    c_tilde = model(X)
    loss = loss_fn(c_true, c_tilde, Sigma)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()