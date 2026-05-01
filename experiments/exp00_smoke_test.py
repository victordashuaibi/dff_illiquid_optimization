"""Smoke test: verify environment is correctly set up."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import xgboost
import cvxpy
import yfinance as yf

from src.dff.bias_correction import BiasCorrectionLayer
from src.losses.regret import normalized_decision_regret
from src.utils.seed import set_seed


def main():
    set_seed(42)
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print(f"XGBoost: {xgboost.__version__}")
    print(f"CVXPY:   {cvxpy.__version__}")

    layer = BiasCorrectionLayer(input_dim=10, output_dim=5, epsilon=0.3)
    x = torch.randn(4, 10)
    c_hat = torch.randn(4, 5).abs() + 0.1
    c_tilde = layer(x, c_hat)

    ratio = (c_tilde / c_hat).abs()
    assert (ratio >= 1 - 0.3 - 1e-5).all() and (ratio <= 1 + 0.3 + 1e-5).all()
    print(f"BiasCorrectionLayer OK | ratio range: [{ratio.min():.3f}, {ratio.max():.3f}]")

    try:
        data = yf.download("SPY", start="2024-01-01", end="2024-02-01", progress=False)
        print(f"yfinance OK | downloaded {len(data)} rows of SPY")
    except Exception as e:
        print(f"yfinance test skipped (network issue): {e}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
