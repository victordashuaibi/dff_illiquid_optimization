"""Panel-style XGBoost backbone for the Two-stage baseline.

A *single* global XGBRegressor is fit on (per-asset feature row,
per-asset return) pairs flattened across all training instances. At
predict time, the regressor is applied row-wise to each instance's
``X`` (shape ``[n_assets, n_features_per_asset]``) and stacked.

This mirrors the panel regression used in the legacy pipeline while
respecting the new instance-format contract from ``docs/INTERFACE.md``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from xgboost import XGBRegressor

from src.data.loader import Instance


# Hyperparameters validated by the legacy stock_xgb_baseline.py script.
DEFAULT_XGB_KWARGS: dict = dict(
    objective="reg:squarederror",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


class XGBoostBackbone:
    """Single-model panel-style regressor (see module docstring)."""

    def __init__(self, **xgb_kwargs):
        params = {**DEFAULT_XGB_KWARGS, **xgb_kwargs}
        self.model: XGBRegressor = XGBRegressor(**params)
        self._fitted: bool = False
        self._n_features: Optional[int] = None

    @staticmethod
    def _stack_panel(instances: list[Instance]) -> tuple[np.ndarray, np.ndarray]:
        """Flatten ``list[Instance]`` into ``(X_panel, c_panel)`` for fitting."""
        if not instances:
            raise ValueError("XGBoostBackbone requires at least one instance.")
        X_panel = np.concatenate([inst.X for inst in instances], axis=0)
        c_panel = np.concatenate([inst.c_true for inst in instances], axis=0)
        return X_panel, c_panel

    def fit(self, instances: list[Instance]) -> None:
        X_panel, c_panel = self._stack_panel(instances)
        self.model.fit(X_panel, c_panel)
        self._fitted = True
        self._n_features = X_panel.shape[1]

    def predict(self, instances: list[Instance]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("XGBoostBackbone.predict called before fit().")
        if not instances:
            return np.empty((0, 0))

        n_instances = len(instances)
        n_assets = instances[0].X.shape[0]
        out = np.empty((n_instances, n_assets), dtype=float)
        for i, inst in enumerate(instances):
            if inst.X.shape[1] != self._n_features:
                raise ValueError(
                    f"Instance {i} has feature dim {inst.X.shape[1]}, "
                    f"expected {self._n_features}"
                )
            out[i] = self.model.predict(inst.X)
        return out
