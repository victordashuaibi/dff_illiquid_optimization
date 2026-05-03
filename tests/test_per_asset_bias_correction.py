"""Tests for ``PerAssetBiasCorrectionLayer`` (decision D1b).

Pinned-down contract:

* per-asset elementwise trust region ``|c_tilde - c_hat| / |c_hat| <= ε``;
* permutation-equivariance over the asset axis;
* shape correctness for arbitrary ``(B, N)``;
* shape rejection for misshaped inputs;
* the NN input is augmented with the per-asset ``c_hat_i`` scalar.
"""
from __future__ import annotations

import pytest
import torch

from src.dff.bias_correction import (
    BiasCorrectionLayer,
    PerAssetBiasCorrectionLayer,
)


def _make_inputs(B: int, N: int, F: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, N, F, generator=g)
    c_hat = torch.randn(B, N, generator=g).abs() + 0.1  # nonzero
    return X, c_hat


def test_per_asset_trust_region_elementwise():
    eps = 0.3
    layer = PerAssetBiasCorrectionLayer(
        n_features_per_asset=4, epsilon=eps, hidden_dim=8, n_layers=3
    )
    X, c_hat = _make_inputs(B=8, N=5, F=4, seed=42)
    c_tilde = layer(X, c_hat)
    assert c_tilde.shape == (8, 5)
    # |c_tilde - c_hat| / |c_hat| <= eps elementwise.
    violation = ((c_tilde - c_hat).abs() / c_hat.abs() - eps).clamp(min=0)
    assert violation.max().item() < 1e-5


def test_per_asset_permutation_equivariance():
    """Shuffle assets along dim=1 → output shuffles identically."""
    layer = PerAssetBiasCorrectionLayer(
        n_features_per_asset=3, epsilon=0.5, hidden_dim=16, n_layers=3
    )
    layer.eval()
    X, c_hat = _make_inputs(B=2, N=4, F=3, seed=7)

    out_orig = layer(X, c_hat)
    perm = torch.tensor([3, 1, 0, 2])
    X_perm = X[:, perm, :]
    c_hat_perm = c_hat[:, perm]
    out_perm = layer(X_perm, c_hat_perm)

    assert torch.allclose(out_perm, out_orig[:, perm], atol=1e-6), (
        "PerAssetBiasCorrectionLayer is not permutation-equivariant — "
        "shared-weight assumption broken"
    )


def test_per_asset_shapes_various_b_n():
    layer = PerAssetBiasCorrectionLayer(
        n_features_per_asset=2, epsilon=0.2, hidden_dim=8, n_layers=2
    )
    for B, N in [(1, 1), (1, 10), (32, 5), (4, 30)]:
        X, c_hat = _make_inputs(B, N, F=2, seed=B * 100 + N)
        out = layer(X, c_hat)
        assert out.shape == (B, N)
        assert torch.isfinite(out).all()


def test_per_asset_rejects_wrong_dim():
    layer = PerAssetBiasCorrectionLayer(
        n_features_per_asset=4, epsilon=0.3, hidden_dim=8
    )
    X = torch.randn(8, 5, 4)
    c_hat = torch.randn(8, 5).abs() + 0.1

    with pytest.raises(ValueError, match="X must have shape"):
        layer(X.reshape(8 * 5, 4), c_hat)
    with pytest.raises(ValueError, match="c_hat must have shape"):
        layer(X, c_hat.flatten())
    with pytest.raises(ValueError, match=r"X feature dim"):
        bad_X = torch.randn(8, 5, 7)  # F=7 vs configured F=4
        layer(bad_X, c_hat)


def test_per_asset_inner_nn_input_dim_includes_c_hat_scalar():
    """The wrapper must build its inner NN with input_dim = F + 1."""
    F = 6
    layer = PerAssetBiasCorrectionLayer(n_features_per_asset=F, epsilon=0.5)
    inner: BiasCorrectionLayer = layer._inner  # implementation handle
    # The first linear layer's in_features is the NN's input dim.
    first_linear = next(m for m in inner.h if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == F + 1, (
        f"inner NN input_dim is {first_linear.in_features}, expected F+1={F + 1} "
        "— the c_hat scalar must be concatenated to the per-asset feature row "
        "(decision D1b in docs/exp02_design.md)"
    )


def test_per_asset_eps_zero_returns_c_hat():
    """ε=0 ⇒ φ = 1 ⇒ c_tilde = c_hat exactly."""
    layer = PerAssetBiasCorrectionLayer(
        n_features_per_asset=3, epsilon=0.0, hidden_dim=4
    )
    X, c_hat = _make_inputs(B=4, N=3, F=3, seed=0)
    c_tilde = layer(X, c_hat)
    assert torch.allclose(c_tilde, c_hat, atol=1e-7)


def test_per_asset_gradient_flows():
    layer = PerAssetBiasCorrectionLayer(
        n_features_per_asset=2, epsilon=0.4, hidden_dim=8
    ).double()
    X = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=False)
    c_hat = torch.randn(3, 4, dtype=torch.float64).abs() + 0.1
    c_hat = c_hat.detach().clone().requires_grad_(False)
    c_tilde = layer(X, c_hat)
    c_tilde.sum().backward()
    # Layer parameters should have nonzero, finite gradients.
    saw_grad = False
    for p in layer.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            saw_grad = saw_grad or (p.grad.abs().sum() > 0).item()
    assert saw_grad, "no nonzero gradient on any wrapper parameter"
