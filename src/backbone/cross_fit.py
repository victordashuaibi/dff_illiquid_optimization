"""Time-respecting k-fold cross-fitting per Chernozhukov et al. (2018).

Given a list of :class:`Instance` objects (one cross-section per trading
day, possibly with date ties in pathological setups), produce
out-of-fold predictions by:

1. Grouping instances by ``metadata["date"]``;
2. Splitting the *unique sorted dates* into ``n_splits`` contiguous
   temporal blocks;
3. For each block ``k``: fit ``backbone_cls`` on the panel-flattened union
   of all blocks ``≠ k`` and predict on block ``k``.

This is **not** ``KFold`` and **not** ``TimeSeriesSplit``. It is the
specific 2-fold scheme DFF (Yang et al. 2025) uses to avoid the backbone
overfitting to in-sample residuals before its predictions are passed to
the bias correction layer.

Backbone API
------------
``backbone_cls(**backbone_kwargs)`` must produce an estimator that
exposes the standard scikit-learn ``fit(X, y)`` / ``predict(X)`` pair on
2D ``X`` and 1D ``y`` (e.g., ``XGBRegressor``, ``KNeighborsRegressor``,
``DummyRegressor``). The helper handles the panel flattening internally
(see :func:`_panel`). Do **not** pass the project's ``XGBoostBackbone``
wrapper here — its ``fit/predict`` signature is over ``list[Instance]``,
not ``(X, y)``. Call this helper with the raw scikit-learn-style class.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Type

import numpy as np
import pandas as pd

from src.data.loader import Instance

logger = logging.getLogger(__name__)


def _panel(instances: list[Instance], positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a slice of ``instances`` (selected by ``positions``) into
    panel-style ``(X_panel, y_panel)``.

    For each selected instance, every asset row contributes one (row of X,
    scalar y) pair. The resulting ``X_panel`` has shape
    ``[sum_p n_assets_p, n_features]`` and ``y_panel`` has shape
    ``[sum_p n_assets_p]``. Same flattening as
    :meth:`src.backbone.xgb.XGBoostBackbone._stack_panel`.
    """
    X_panel = np.concatenate([instances[p].X for p in positions], axis=0)
    y_panel = np.concatenate([instances[p].c_true for p in positions], axis=0)
    return X_panel, y_panel


def cross_fit_predict(
    backbone_cls: Type,
    backbone_kwargs: dict[str, Any],
    instances: list[Instance],
    n_splits: int = 2,
) -> np.ndarray:
    """
    Time-respecting k-fold cross-fitting per Chernozhukov et al. (2018).

    Sort instances by ``metadata["date"]`` ascending. Split into
    ``n_splits`` contiguous temporal blocks. For each fold k:
        - train backbone on the union of all blocks ≠ k
        - predict on block k (out-of-fold)
    Return predictions concatenated back into the *original* instance order
    (not date-sorted order). Caller passes instances in some order; caller
    gets predictions back in that same order.

    NOT a random k-fold. NOT shuffled. NOT stratified. The temporal block
    structure is what makes this honest under autocorrelated time series.

    Date ties (multiple instances sharing one ``metadata["date"]``, e.g.
    in synthetic multi-asset fixtures) are kept in the same fold; date
    groups are atomic.

    Parameters
    ----------
    backbone_cls
        A scikit-learn-style estimator class. ``backbone_cls(**backbone_kwargs)``
        must produce an object with ``fit(X, y)`` and ``predict(X)``.
    backbone_kwargs
        Keyword arguments forwarded to ``backbone_cls``.
    instances
        Input cross-section list. Order is preserved on return.
    n_splits
        Number of temporal folds (paper default: 2).

    Returns
    -------
    np.ndarray of shape ``(len(instances), n_assets)``
        ``out[i]`` is the OOF prediction for ``instances[i]`` in the
        caller's original ordering.

    Raises
    ------
    ValueError
        If ``instances`` is empty, ``n_splits < 2``, or ``n_splits``
        exceeds the number of unique decision dates.
    """
    if not instances:
        raise ValueError("instances must be non-empty")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    n_instances = len(instances)
    n_assets = instances[0].X.shape[0]

    # Per-instance dates and their original positions.
    dates = [pd.Timestamp(inst.metadata["date"]) for inst in instances]

    # Group original positions by date so same-date instances are atomic.
    date_to_positions: dict[pd.Timestamp, list[int]] = defaultdict(list)
    for pos, d in enumerate(dates):
        date_to_positions[d].append(pos)
    sorted_dates: list[pd.Timestamp] = sorted(date_to_positions.keys())
    n_dates = len(sorted_dates)

    if n_dates < n_splits:
        raise ValueError(
            f"n_splits={n_splits} > number of unique dates ({n_dates}); "
            f"cannot assign each fold a non-empty contiguous block"
        )

    # Contiguous temporal blocks of date *indices*. np.array_split balances
    # block sizes within 1 when n_dates is not a multiple of n_splits.
    fold_date_idx_blocks: list[np.ndarray] = np.array_split(np.arange(n_dates), n_splits)

    output = np.empty((n_instances, n_assets), dtype=float)

    for k in range(n_splits):
        train_date_idxs = np.concatenate(
            [fold_date_idx_blocks[j] for j in range(n_splits) if j != k]
        )
        test_date_idxs = fold_date_idx_blocks[k]

        train_positions: list[int] = []
        for di in train_date_idxs:
            train_positions.extend(date_to_positions[sorted_dates[di]])
        test_positions: list[int] = []
        for di in test_date_idxs:
            test_positions.extend(date_to_positions[sorted_dates[di]])

        X_train, y_train = _panel(instances, train_positions)

        train_dates_in_fold = [dates[p] for p in train_positions]
        test_dates_in_fold = [dates[p] for p in test_positions]
        logger.info(
            "fold %d/%d: train=%d instances, predict=%d instances, "
            "train dates [%s, %s], predict dates [%s, %s]",
            k + 1,
            n_splits,
            len(train_positions),
            len(test_positions),
            min(train_dates_in_fold).date(),
            max(train_dates_in_fold).date(),
            min(test_dates_in_fold).date(),
            max(test_dates_in_fold).date(),
        )

        # Train, predict, discard. One model in memory at a time.
        model = backbone_cls(**backbone_kwargs)
        model.fit(X_train, y_train)
        for p in test_positions:
            preds = model.predict(instances[p].X)
            preds = np.asarray(preds, dtype=float).reshape(-1)
            if preds.shape[0] != n_assets:
                raise RuntimeError(
                    f"backbone returned {preds.shape[0]} preds for instance {p} "
                    f"(expected n_assets={n_assets})"
                )
            output[p] = preds
        del model

    return output
