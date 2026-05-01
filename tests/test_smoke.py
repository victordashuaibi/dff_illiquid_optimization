"""Smoke tests — verify imports and basic forward passes for each src module."""
import torch
import pytest


def test_imports():
    from src.utils.seed import set_seed
    from src.losses.regret import decision_regret, normalized_decision_regret
    from src.dff.bias_correction import BiasCorrectionLayer


def test_set_seed():
    from src.utils.seed import set_seed
    set_seed(0)
    a = torch.randn(3)
    set_seed(0)
    b = torch.randn(3)
    assert torch.allclose(a, b)


def test_decision_regret_zero_when_optimal():
    from src.losses.regret import decision_regret
    c = torch.tensor([1.0, 2.0, 3.0])
    w = torch.tensor([0.5, 0.3, 0.2])
    dr = decision_regret(c, w, w)
    assert dr.item() == pytest.approx(0.0)


def test_decision_regret_nonnegative_for_worse_solution():
    from src.losses.regret import decision_regret
    c = torch.tensor([1.0, 1.0])
    w_oracle = torch.tensor([0.0, 1.0])  # higher c^T w
    w_pred = torch.tensor([1.0, 0.0])    # lower c^T w
    dr = decision_regret(c, w_pred, w_oracle)
    assert dr.item() == pytest.approx(0.0)  # equal objective here


def test_ndr_shape():
    from src.losses.regret import normalized_decision_regret
    B = 8
    n = 5
    c = torch.randn(B, n)
    w = torch.randn(B, n)
    ndr = normalized_decision_regret(c, w, w)
    assert ndr.shape == torch.Size([])  # scalar
    assert ndr.item() == pytest.approx(0.0)


def test_bias_correction_output_range():
    from src.dff.bias_correction import BiasCorrectionLayer
    eps = 0.3
    layer = BiasCorrectionLayer(input_dim=10, output_dim=5, epsilon=eps)
    x = torch.randn(4, 10)
    c_hat = torch.ones(4, 5)
    c_tilde = layer(x, c_hat)
    assert c_tilde.shape == (4, 5)
    # phi must lie in [1-eps, 1+eps] so c_tilde/c_hat in that range
    ratio = c_tilde / c_hat
    assert (ratio >= 1 - eps - 1e-6).all()
    assert (ratio <= 1 + eps + 1e-6).all()


def test_bias_correction_trust_region():
    """Eq. 10: |c_tilde - c_hat| / |c_hat| <= eps element-wise."""
    from src.dff.bias_correction import BiasCorrectionLayer
    eps = 0.3
    layer = BiasCorrectionLayer(input_dim=6, output_dim=4, epsilon=eps)
    x = torch.randn(16, 6)
    c_hat = torch.randn(16, 4).abs() + 0.1  # keep nonzero
    c_tilde = layer(x, c_hat)
    violation = ((c_tilde - c_hat).abs() / c_hat.abs() - eps).clamp(min=0)
    assert violation.max().item() < 1e-5
