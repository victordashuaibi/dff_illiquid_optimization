# DFF Project Interface Contract

This document defines the interfaces that both teammates must adhere to.
Any change to this contract requires discussion and joint agreement.

## Data Instance Format

All data flowing through the pipeline uses this format:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Instance:
    X: np.ndarray         # features, shape [n_features]
    c_true: np.ndarray    # ground truth returns, shape [n_assets]
    Sigma: np.ndarray     # covariance matrix, shape [n_assets, n_assets]
    metadata: dict        # {date, instance_id, ...}
```

A batch of instances is represented as `list[Instance]` or, for tensor
operations, as separate stacked arrays:
- X_batch: [batch_size, n_features]
- c_batch: [batch_size, n_assets]
- Sigma:   [n_assets, n_assets]  (shared across batch by default)

## Backbone Interface (Victor)

Located at `src/backbone/`.

```python
class Backbone:
    def fit(self, instances: list[Instance]) -> None:
        """Train on a list of instances using MSE loss."""
        ...

    def predict(self, instances: list[Instance]) -> np.ndarray:
        """Returns c_hat of shape [n_instances, n_assets]."""
        ...
```

Default implementation: `XGBoostBackbone` in `src/backbone/xgb.py`.

## Differentiable Optimizer Interface (Teammate)

Located at `src/optimizer/markowitz_diff.py`.

```python
import torch
import torch.nn as nn

class DiffMarkowitz(nn.Module):
    def __init__(self, n_assets: int, risk_aversion: float = 1.0,
                 long_only: bool = True):
        ...

    def forward(
        self,
        c_hat: torch.Tensor,    # [batch, n_assets], requires_grad=True
        Sigma: torch.Tensor,    # [n_assets, n_assets]
    ) -> torch.Tensor:
        """Returns w: [batch, n_assets], requires_grad=True.
        Constraints: sum(w) = 1, w >= 0 (if long_only).
        """
        ...
```

Static (non-differentiable) version: `MarkowitzStatic` in
`src/optimizer/markowitz_static.py` (Victor writes this for baseline).

## Markowitz Mathematical Form (Both Teammates Must Match)

Both `MarkowitzStatic` and `DiffMarkowitz` solve the same problem:

```
minimize  -c^T w + gamma * w^T Sigma w
s.t.      sum(w) = 1
          w >= 0   (if long_only=True)
```

Default: `gamma = 1.0`, `long_only = True`.

## Covariance Matrix Estimation

Estimated using Ledoit-Wolf shrinkage from `sklearn.covariance.LedoitWolf`.
Computed once per instance (or rolling window) by the data loader.

## Decision Regret and NDR

Located at `src/losses/regret.py` (already implemented).

```python
def normalized_decision_regret(
    c_true: torch.Tensor,    # [batch, n_assets]
    w_pred: torch.Tensor,    # [batch, n_assets]
    w_oracle: torch.Tensor,  # [batch, n_assets], from oracle solver
) -> torch.Tensor:
    """NDR per Tang & Khalil 2022 (Eq. 19 of DFF paper)."""
```

## SPO+ Loss

Located at `src/losses/spo_plus.py` (teammate writes).

Implements DFF paper Eq. 17:
```
L_SPO+ = min_w {(2*c_tilde - c)^T w} + 2*c_tilde^T w*(c) - f*(c)
```

## Bias Correction Layer (Already Implemented)

Located at `src/dff/bias_correction.py`.

Implements DFF paper Eq. 9-11:
```
F_theta(x) = phi(x) * c_hat
phi(x) = (1 - epsilon) + 2*epsilon * sigmoid(h(x))
```

Trust region constraint: |c_tilde - c_hat| / |c_hat| <= epsilon.

## Sanity Test (Run at Week 2 Sync Point)

Both teammates run this jointly to verify the optimizers agree:

```python
import torch
import numpy as np
from src.optimizer.markowitz_static import MarkowitzStatic
from src.optimizer.markowitz_diff import DiffMarkowitz

n_assets = 10
torch.manual_seed(42)
c = torch.randn(1, n_assets) * 0.01
Sigma_raw = torch.randn(n_assets, n_assets)
Sigma = Sigma_raw @ Sigma_raw.T + 0.01 * torch.eye(n_assets)  # PSD

w_static = MarkowitzStatic(n_assets).solve(c.numpy().squeeze(), Sigma.numpy())
w_diff = DiffMarkowitz(n_assets)(c, Sigma).detach().numpy().squeeze()

assert np.max(np.abs(w_static - w_diff)) < 1e-4, "Optimizers disagree!"
```

## Default Dimensions for Synthetic Testing

- `n_assets = 30`
- `n_samples = 1000`
- `n_features = 40` (matches Victor's feature engineering)

## Change Process

Any change to interfaces in this document:
1. Open an issue on GitHub describing the proposed change
2. Both teammates discuss
3. Update this document FIRST
4. Then update code to match
