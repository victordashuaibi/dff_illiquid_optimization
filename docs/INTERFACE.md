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
    X: np.ndarray         # features, shape [n_assets, n_features_per_asset]
    c_true: np.ndarray    # ground truth returns, shape [n_assets]
    Sigma: np.ndarray     # covariance matrix, shape [n_assets, n_assets]
    metadata: dict        # {date, ticker_list, instance_id, ...}
```

`X` is 2D so a single global panel-style backbone can be trained on
`(per-asset features, per-asset return)` pairs while still emitting a
`c_hat` vector of shape `[n_assets]` per instance.

A batch of instances is represented as `list[Instance]` or, for tensor
operations, as separate stacked arrays:
- X_batch: [batch_size, n_assets, n_features_per_asset]
- c_batch: [batch_size, n_assets]
- Sigma:   [n_assets, n_assets]  (shared across batch by default)

## Instance Invariants

1. **Asset ordering must be consistent across all instances in a dataset.**
   For any two instances `inst_a`, `inst_b` in the same `list[Instance]`:
   - `inst_a.metadata['ticker_list'] == inst_b.metadata['ticker_list']`
   - `inst_a.X.shape[0] == inst_b.X.shape[0] == n_assets`
   - `inst_a.c_true.shape[0] == inst_b.c_true.shape[0] == n_assets`

   This means `c_true[i]`, `X[i, :]`, and `Sigma[i, j]` all refer to the
   same asset (the `i`-th ticker in `ticker_list`).

2. **Sigma is symmetric and positive semi-definite.** Verify with:
   ```python
   assert np.allclose(Sigma, Sigma.T)
   assert np.linalg.eigvalsh(Sigma).min() >= -1e-8
   ```

3. **No NaN allowed.** If any ticker has NaN at time `t`, drop the
   entire instance (do not impute).

`PortfolioDataLoader.load()` must enforce these three invariants and run a
sanity check before yielding instances; assertion failures should report
which instance (date) and which asset triggered the violation.

## Backbone Interface (Victor)

Located at `src/backbone/`.

```python
class Backbone:
    def fit(self, instances: list[Instance]) -> None:
        """Train on a list of instances using MSE loss.

        The default XGBoostBackbone is *panel-style*: a single global
        regressor is fit on all (per-asset feature row, per-asset return)
        pairs flattened across instances. Concretely, training data is
            X_panel = np.concatenate([inst.X for inst in instances], axis=0)
                      # shape [n_instances * n_assets, n_features_per_asset]
            c_panel = np.concatenate([inst.c_true for inst in instances])
                      # shape [n_instances * n_assets]
        which mirrors the panel regression used in the Two-stage baseline.
        """
        ...

    def predict(self, instances: list[Instance]) -> np.ndarray:
        """Returns c_hat of shape [n_instances, n_assets].

        For each instance, the global regressor is applied row-wise to
        `instance.X` (shape [n_assets, n_features_per_asset]) and stacked.
        """
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

Located at `src/losses/regret.py`.

### Two NDR variants

There are two NDR functions because we deal with two types of objectives:

1. **Linear** (used by SPO+ on shortest path / matching / Wilder 2019 toy):
   - Use `normalized_decision_regret(c_true, w_pred, w_oracle)`
   - `f(w, c) = c^T w` (minimization)

2. **Quadratic** (used by Markowitz mean-variance — every portfolio
   experiment in this project):
   - Use `markowitz_normalized_decision_regret(c_true, w_pred, w_oracle, Sigma, risk_aversion)`
   - `f(w, c, Σ) = -c^T w + γ w^T Σ w`
   - Required because using the linear-only formula with our Markowitz
     setup yields negative or wrong-signed regret (the oracle trades return
     for risk reduction, so `c^T w_oracle` is *not* the maximum of
     `c^T w` taken over the simplex).

Both functions implement DFF paper Eq. 19:
```
NDR = sum_b (f(w_pred_b) - f(w_oracle_b)) / sum_b |f(w_oracle_b)|
```
The numerator is non-negative (oracle is optimal); the denominator uses
the oracle objective absolute value.

```python
# Linear variant (for SPO+, shortest path, etc.)
def normalized_decision_regret(
    c_true: torch.Tensor,    # [batch, n_assets]
    w_pred: torch.Tensor,    # [batch, n_assets]
    w_oracle: torch.Tensor,  # [batch, n_assets]
) -> torch.Tensor: ...

# Quadratic variant (for Markowitz mean-variance)
def markowitz_normalized_decision_regret(
    c_true: torch.Tensor,    # [batch, n_assets]
    w_pred: torch.Tensor,    # [batch, n_assets]
    w_oracle: torch.Tensor,  # [batch, n_assets]
    Sigma: torch.Tensor,     # [n_assets, n_assets] or [batch, n_assets, n_assets]
    risk_aversion: float = 1.0,
) -> torch.Tensor: ...
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
