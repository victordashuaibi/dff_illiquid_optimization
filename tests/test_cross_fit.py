"""Tests for ``src.backbone.cross_fit.cross_fit_predict``.

The contract these tests pin down:

* every input instance gets exactly one out-of-fold prediction;
* fold boundaries are date-monotonic (contiguous temporal blocks);
* the output array is in *caller's* original input order, not date-sorted;
* OOF predictions are actually OOF — a backbone that memorizes its
  training set must NOT recover the training labels;
* multi-asset fixtures with date ties keep same-date instances in the
  same fold;
* ``n_splits=2`` is the paper-default.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.backbone.cross_fit import cross_fit_predict
from src.data.loader import Instance


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _instance(
    date: str,
    n_assets: int = 4,
    n_features: int = 3,
    seed: int = 0,
    c_true: Optional[np.ndarray] = None,
) -> Instance:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_assets, n_features))
    if c_true is None:
        c_true = rng.normal(size=n_assets) * 0.05
    Sigma = np.eye(n_assets) * 0.01
    return Instance(
        X=X, c_true=c_true, Sigma=Sigma,
        metadata={"date": pd.Timestamp(date),
                  "ticker_list": [f"T{i:02d}" for i in range(n_assets)]},
    )


def _make_synthetic_panel(
    n_dates: int = 12,
    n_assets: int = 5,
    n_features: int = 4,
    start: str = "2020-01-02",
    noise: float = 0.05,
    seed: int = 0,
) -> list[Instance]:
    """Synthetic panel where ``c_true = β·X + noise`` so a real model has
    something to learn but a memorizer can still distinguish in-sample
    from out-of-sample.
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=n_features) * 0.1
    dates = pd.bdate_range(start, periods=n_dates)
    instances: list[Instance] = []
    for d in dates:
        X = rng.normal(size=(n_assets, n_features))
        c_true = X @ beta + noise * rng.normal(size=n_assets)
        Sigma = np.eye(n_assets) * 0.01
        instances.append(Instance(
            X=X, c_true=c_true, Sigma=Sigma,
            metadata={"date": pd.Timestamp(d),
                      "ticker_list": [f"T{i:02d}" for i in range(n_assets)]},
        ))
    return instances


# ---------------------------------------------------------------------------
# 1. each instance has exactly one OOF prediction; output shape matches input
# ---------------------------------------------------------------------------
def test_each_instance_predicted_exactly_once():
    instances = _make_synthetic_panel(n_dates=10, n_assets=4)
    preds = cross_fit_predict(
        DummyRegressor, {"strategy": "mean"}, instances, n_splits=2
    )
    assert preds.shape == (len(instances), 4)
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------------
# 2. fold boundaries are date-monotonic
# ---------------------------------------------------------------------------
def test_fold_boundaries_are_date_monotonic():
    """Build c_true that is strictly monotone in date so each fold's
    ``DummyRegressor("mean")`` outputs a fold-distinguishable constant.
    Then recover fold assignment from predictions and assert dates in
    each fold are an interval that doesn't overlap with the others.
    """
    n_dates = 12
    n_assets = 3
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    instances: list[Instance] = []
    for t, d in enumerate(dates):
        X = rng.normal(size=(n_assets, 2))
        c_true = np.full(n_assets, float(t) * 1.0)  # date-index encoded
        Sigma = np.eye(n_assets) * 0.01
        instances.append(Instance(
            X=X, c_true=c_true, Sigma=Sigma,
            metadata={"date": pd.Timestamp(d),
                      "ticker_list": [f"T{i:02d}" for i in range(n_assets)]},
        ))

    preds = cross_fit_predict(
        DummyRegressor, {"strategy": "mean"}, instances, n_splits=3
    )

    # Each instance's prediction is a constant vector of size n_assets equal
    # to the train fold's c_true mean. Different folds have different train
    # means, so this scalar uniquely tags the fold.
    fold_tag = preds[:, 0]
    instance_dates = np.array([pd.Timestamp(i.metadata["date"]) for i in instances])

    unique_tags = np.unique(np.round(fold_tag, 8))
    assert len(unique_tags) == 3, (
        f"expected 3 distinct fold tags for 3-fold split, got {len(unique_tags)}"
    )

    fold_id_per_instance = np.searchsorted(np.sort(unique_tags), np.round(fold_tag, 8))
    # The fold whose train-mean is *highest* corresponds to the *earliest* test
    # block (because the early block is held out, so training mean is high).
    # We don't assume which sort direction; we just check that each fold's
    # date-range is contiguous and non-overlapping with the others.
    for fid in range(3):
        in_fold = instance_dates[fold_id_per_instance == fid]
        in_fold_sorted = np.sort(in_fold)
        # contiguous in date — the dates assigned to a fold form a prefix of
        # sorted_dates from some start to some end with no gaps relative to
        # the global date list.
        assert len(in_fold_sorted) > 0
        global_idx = np.searchsorted(np.sort(np.unique(instance_dates)), in_fold_sorted)
        assert np.all(np.diff(global_idx) == 1), (
            f"fold {fid} dates {in_fold_sorted} are not a contiguous block "
            f"(global indices {global_idx})"
        )

    # Folds are non-overlapping (already implied by len(unique_tags)==3 and
    # each instance having one fold tag, but assert explicitly).
    fold_date_sets = [set(instance_dates[fold_id_per_instance == fid]) for fid in range(3)]
    for a in range(3):
        for b in range(a + 1, 3):
            assert fold_date_sets[a].isdisjoint(fold_date_sets[b])


# ---------------------------------------------------------------------------
# 3. predictions returned in original input order
# ---------------------------------------------------------------------------
def test_predictions_returned_in_original_input_order():
    """Shuffle inputs two different ways, run cross_fit_predict on each,
    re-align by date. Per-date predictions must agree across the two runs.
    """
    instances = _make_synthetic_panel(n_dates=10, n_assets=4, seed=1)

    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(222)
    perm_a = rng_a.permutation(len(instances))
    perm_b = rng_b.permutation(len(instances))
    inst_a = [instances[i] for i in perm_a]
    inst_b = [instances[i] for i in perm_b]

    preds_a = cross_fit_predict(KNeighborsRegressor, {"n_neighbors": 1}, inst_a)
    preds_b = cross_fit_predict(KNeighborsRegressor, {"n_neighbors": 1}, inst_b)

    # Re-align: for each canonical position i in the original list, find where
    # it ended up in each shuffled run, and compare.
    inv_a = np.argsort(perm_a)
    inv_b = np.argsort(perm_b)
    aligned_a = preds_a[inv_a]
    aligned_b = preds_b[inv_b]
    assert np.allclose(aligned_a, aligned_b, atol=1e-8), (
        "cross_fit_predict gave different aligned-by-date predictions for two "
        "input orderings — order preservation is broken"
    )


# ---------------------------------------------------------------------------
# 4. critical: OOF is actually out-of-fold (memorizer test)
# ---------------------------------------------------------------------------
def test_oof_is_actually_out_of_fold():
    """KNN(k=1) memorizes training rows. In-sample predictions on training
    rows recover ``y_train`` exactly. OOF predictions cannot — they have
    to fall back to the nearest *other* training row, which generally
    differs from the held-out true label.
    """
    instances = _make_synthetic_panel(
        n_dates=20, n_assets=4, n_features=4, noise=0.1, seed=42
    )
    n_assets = instances[0].X.shape[0]

    # In-sample: train KNN(k=1) on all data, predict on all data.
    X_all = np.concatenate([inst.X for inst in instances], axis=0)
    y_all = np.concatenate([inst.c_true for inst in instances], axis=0)
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_all, y_all)
    y_in_sample = knn.predict(X_all).reshape(len(instances), n_assets)
    y_true = np.stack([inst.c_true for inst in instances], axis=0)
    assert np.allclose(y_in_sample, y_true, atol=1e-8), (
        "sanity: KNN(k=1) must memorize training labels exactly"
    )

    # OOF: cross_fit_predict with the same backbone class.
    y_oof = cross_fit_predict(
        KNeighborsRegressor, {"n_neighbors": 1}, instances, n_splits=2
    )
    assert y_oof.shape == y_true.shape

    diff = np.abs(y_oof - y_true).mean()
    threshold = 0.01 * float(np.std(y_true))
    assert diff > threshold, (
        f"OOF predictions look identical to ground truth "
        f"(mean abs diff {diff:.6e} <= {threshold:.6e}); "
        f"fold assignment is broken — instances are predicting themselves"
    )


# ---------------------------------------------------------------------------
# 5. same-date instances share fold
# ---------------------------------------------------------------------------
def test_same_date_instances_share_fold():
    """Multi-asset / date-tie fixture: build several instances per date
    and confirm that within a date, every instance receives the same OOF
    prediction (i.e., they were all trained on the same fold's data and
    a feature-agnostic backbone gave them the same constant).
    """
    rng = np.random.default_rng(7)
    instances: list[Instance] = []
    # Two instances per date for 8 dates -> 16 instances total.
    dates = pd.bdate_range("2020-03-02", periods=8)
    for d in dates:
        for replica in range(2):
            n_assets = 3
            X = rng.normal(size=(n_assets, 2))
            # c_true depends only on date so DummyRegressor("mean") becomes
            # a non-trivial fold-distinguishable predictor.
            c_true = np.full(n_assets, float(d.value))
            Sigma = np.eye(n_assets) * 0.01
            instances.append(Instance(
                X=X, c_true=c_true, Sigma=Sigma,
                metadata={"date": pd.Timestamp(d),
                          "ticker_list": [f"T{i:02d}" for i in range(n_assets)],
                          "replica": replica},
            ))

    preds = cross_fit_predict(
        DummyRegressor, {"strategy": "mean"}, instances, n_splits=2
    )

    # Group predictions by date; within a date, all instances should have
    # *identical* prediction vectors (DummyRegressor("mean") is a constant
    # function of the train data, and same-date instances share the train
    # data because they're in the same fold).
    by_date: dict[pd.Timestamp, list[np.ndarray]] = {}
    for inst, pred in zip(instances, preds):
        by_date.setdefault(pd.Timestamp(inst.metadata["date"]), []).append(pred)

    for d, vec_list in by_date.items():
        ref = vec_list[0]
        for v in vec_list[1:]:
            assert np.allclose(v, ref, atol=1e-12), (
                f"on {d.date()}: same-date instances received different "
                f"predictions ({v} vs {ref}) — same-date split across folds"
            )


# ---------------------------------------------------------------------------
# 6. n_splits=2 is the default (paper)
# ---------------------------------------------------------------------------
def test_n_splits_2_is_default():
    """``cross_fit_predict(...)`` without ``n_splits`` matches the paper's
    2-fold (Chernozhukov et al. 2018, Yang et al. 2025).
    """
    instances = _make_synthetic_panel(n_dates=10, n_assets=3, seed=3)
    preds_default = cross_fit_predict(
        DummyRegressor, {"strategy": "mean"}, instances
    )
    preds_explicit_2 = cross_fit_predict(
        DummyRegressor, {"strategy": "mean"}, instances, n_splits=2
    )
    assert np.allclose(preds_default, preds_explicit_2)


# ---------------------------------------------------------------------------
# 7. error handling
# ---------------------------------------------------------------------------
def test_empty_instances_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cross_fit_predict(DummyRegressor, {"strategy": "mean"}, [])


def test_n_splits_lt_2_raises():
    instances = _make_synthetic_panel(n_dates=4)
    with pytest.raises(ValueError, match="n_splits"):
        cross_fit_predict(DummyRegressor, {"strategy": "mean"}, instances, n_splits=1)


def test_n_splits_gt_n_dates_raises():
    instances = _make_synthetic_panel(n_dates=3)
    with pytest.raises(ValueError, match="unique dates"):
        cross_fit_predict(DummyRegressor, {"strategy": "mean"}, instances, n_splits=5)
