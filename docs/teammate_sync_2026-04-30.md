# Sync notes — 2026-04-30

## What changed on main since you last pulled

1. **Baseline merged (d490df5 → 4414d29)**
   - Two-stage Markowitz baseline pipeline complete
   - XGBoost backbone (panel-style training, single global model)
   - Static Markowitz optimizer (cvxpy, non-differentiable)
   - Mini experiment NDR = 105.93% on 5 mega-caps × 2020-2022

2. **Your `markowitz_diff.py` was patched (6cce214)**
   - Added `from __future__ import annotations` for Python 3.9 compatibility
   - Original code used `dict | None` which is Python 3.10+ syntax
   - Going forward: either keep `from __future__ import annotations` at the
     top of every new file, or use `Optional[X]` from `typing`

3. **`numpy` pinned to `<2.0` (9dcbcd0)**
   - `torch` 2.2 and `xgboost` 2.1 were compiled against numpy 1.x ABI
   - numpy 2.0 broke `torch.Tensor.numpy()` and similar bridges
   - To sync: `pip install -r requirements.txt --upgrade` (downgrades numpy)

4. **`INTERFACE.md` updated (please re-read)**
   - `Instance.X` is now 2D: `[n_assets, n_features_per_asset]`
   - Two NDR variants:
     - `normalized_decision_regret()` — linear objective (your SPO+ tasks)
     - `markowitz_normalized_decision_regret()` — quadratic objective (portfolio)
   - Added an **Instance Invariants** section (asset ordering, PSD Sigma, no NaN)

## To sync your local

```bash
git checkout main
git pull origin main
pip install -r requirements.txt --upgrade
pytest tests/ -v   # should be 24/24
```

## Workflow reminder

Please use feature branches + PR going forward, not direct push to `main`.
I'll enable branch protection on `main` this week.
