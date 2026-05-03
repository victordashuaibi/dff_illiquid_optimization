"""Determinism test for the DFF trainer.

Two back-to-back calls to :func:`run_trainer` with the same config and
instances must produce bit-identical (or float64-tight) per-epoch and
final metrics. Anything else is a leak (unseeded RNG, unstable order of
parallel reductions, hash-randomized iteration, etc.).

The test uses a synthetic 200-instance / 5-asset / 10-feature fixture
spanning two calendar years so the trainer's split / cross-fit / val /
embargo logic still exercises end-to-end. ``epochs=2`` keeps the test
under a few seconds.

Note on XGBoost: with ``n_jobs > 1`` and ``tree_method="hist"``, OpenMP
thread-completion order can introduce tiny non-determinism in tree
splits. The test fixture therefore overrides ``n_jobs=1`` for the XGB
backbone. The production config (``base_config()``) keeps ``n_jobs=-1``
for speed; full multi-thread determinism on real data is best-effort
and not asserted here.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from experiments.exp02_dff_train import base_config, run_trainer
from src.data.loader import Instance


# ---------------------------------------------------------------------------
# Synthetic fixture: 200 instances, 5 assets, 10 features, two calendar years
# ---------------------------------------------------------------------------
def _synth_instances(
    n_per_year: int = 100,
    n_assets: int = 5,
    n_features: int = 10,
    seed: int = 12345,
) -> list[Instance]:
    rng = np.random.default_rng(seed)
    # Two calendar years so PortfolioDataLoader.split(test_year=2021) has
    # a real boundary to embargo around.
    dates = list(pd.bdate_range("2020-01-02", periods=n_per_year)) + \
            list(pd.bdate_range("2021-01-04", periods=n_per_year))
    # Shared signal so M_full has something to learn (otherwise gradients
    # die and the test still passes deterministically — but it's nicer for
    # debugging when ndr improvement is non-trivial).
    beta = rng.normal(size=n_features) * 0.05
    A = rng.normal(size=(n_assets, n_assets))
    Sigma = (A @ A.T) / n_assets + 0.01 * np.eye(n_assets)
    Sigma = 0.5 * (Sigma + Sigma.T)

    instances: list[Instance] = []
    for d in dates:
        X = rng.normal(size=(n_assets, n_features))
        c_true = X @ beta + 0.02 * rng.normal(size=n_assets)
        instances.append(Instance(
            X=X, c_true=c_true, Sigma=Sigma.copy(),
            metadata={"date": pd.Timestamp(d),
                      "ticker_list": [f"T{i:02d}" for i in range(n_assets)],
                      "instance_id": int(np.random.default_rng(int(d.value)).integers(0, 10**9))},
        ))
    return instances


def _test_config() -> dict:
    cfg = base_config()
    # Synthetic-fixture overrides — keep paper-faithful otherwise.
    cfg["test_year"] = 2021
    cfg["embargo_days"] = 21
    cfg["epochs"] = 2
    cfg["val_fraction"] = 0.10
    cfg["seed"] = 4242
    cfg["batch_size"] = 8
    # Force single-threaded XGBoost — multi-thread hist has minor
    # non-determinism due to OpenMP reduction order.
    cfg["xgb_kwargs"] = dict(cfg["xgb_kwargs"])
    cfg["xgb_kwargs"]["n_jobs"] = 1
    cfg["xgb_kwargs"]["random_state"] = cfg["seed"]
    cfg["xgb_kwargs"]["n_estimators"] = 20  # keep test fast
    return cfg


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------
def test_trainer_is_deterministic_across_runs():
    """Two back-to-back runs with the same seed produce identical metrics.

    Uses the synthetic 200-instance fixture, epochs=2, single-threaded
    XGBoost. Asserts:
      - per-epoch dict keys match
      - per-epoch numeric values agree to 1e-12 absolute
      - final test_metrics dict agrees to 1e-12 absolute
    """
    cfg = _test_config()
    instances = _synth_instances()

    res1 = run_trainer(deepcopy(cfg), instances, run_dir=None, logger=None)
    res2 = run_trainer(deepcopy(cfg), instances, run_dir=None, logger=None)

    # ---- per-epoch metrics ----
    em1 = res1["epoch_metrics"]
    em2 = res2["epoch_metrics"]
    assert len(em1) == len(em2) == cfg["epochs"], (
        f"epoch counts differ: run1={len(em1)}, run2={len(em2)}, expected {cfg['epochs']}"
    )
    for i, (r1, r2) in enumerate(zip(em1, em2)):
        assert r1.keys() == r2.keys(), f"epoch {i+1}: key sets differ"
        for k in r1:
            v1, v2 = r1[k], r2[k]
            if isinstance(v1, (int, np.integer)):
                assert v1 == v2, f"epoch {i+1} key={k}: {v1} != {v2}"
            else:
                assert abs(float(v1) - float(v2)) < 1e-12, (
                    f"epoch {i+1} key={k}: {v1} != {v2} (|delta|={abs(v1 - v2):.3e})"
                )

    # ---- final test metrics ----
    f1 = res1["final"]
    f2 = res2["final"]
    assert f1.keys() == f2.keys()
    for k in f1:
        v1, v2 = f1[k], f2[k]
        if isinstance(v1, (int, np.integer)):
            assert v1 == v2, f"final key={k}: {v1} != {v2}"
        else:
            assert abs(float(v1) - float(v2)) < 1e-12, (
                f"final key={k}: {v1} != {v2} (|delta|={abs(v1 - v2):.3e})"
            )


def test_universe_hash_is_stable_for_same_tickers():
    """``base_config()`` yields the same universe_hash for the same ticker set."""
    a = base_config()
    b = base_config()
    assert a["universe_hash"] == b["universe_hash"]
    assert len(a["universe_hash"]) == 8
    # Sorting changes the input list but not the hash (hash is on sorted tuple).
    from experiments.exp02_dff_train import _universe_hash, ILLIQ_30_TICKERS
    shuffled = list(reversed(ILLIQ_30_TICKERS))
    assert _universe_hash(shuffled) == a["universe_hash"]
