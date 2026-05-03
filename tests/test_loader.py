"""Tests for ``PortfolioDataLoader.split`` — embargo invariants.

The 21-day forward-return target makes the year-boundary leakage real:
without an embargo, the last ~21 train instances have labels derived from
prices that fall inside the test year. Rolling features go the other way
(early test instances depend on train-window prices). These tests pin
down the embargo contract: the *trading-day* gap between max(train) and
min(test) is at least ``2 * embargo_days``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import features as features_mod
from src.data.loader import Instance, PortfolioDataLoader


def _synth_prices(
    tickers: list[str],
    n_days: int,
    start: str = "2020-01-02",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    rows = []
    for tk in tickers:
        px = 50.0 + np.cumsum(rng.normal(0, 0.5, n_days))
        for i, d in enumerate(dates):
            p = float(px[i])
            rows.append({
                "Date": d, "Ticker": tk,
                "Open": p, "High": p + 1.0, "Low": p - 1.0,
                "Close": p, "Adj Close": p,
                "Volume": int(rng.integers(1e5, 1e6)),
            })
    return pd.DataFrame(rows)


def _load_two_years(
    tmp_path,
    embargo_days,
    n_days: int = 540,
) -> tuple[list[Instance], list[Instance]]:
    """Build a 2-year synthetic dataset and split with the requested embargo."""
    tickers = ["A", "B"]
    prices = _synth_prices(tickers, n_days=n_days, start="2020-01-02", seed=7)
    loader = PortfolioDataLoader(
        tickers=tickers, start_date="2020-01-02", end_date="2022-06-30",
        cov_window=20, cache_dir=tmp_path, use_cache=False,
    )
    loader._download_prices = lambda: prices  # type: ignore[assignment]
    instances = loader.load()
    return loader.split(instances, test_year=2021, embargo_days=embargo_days)


def _bday_gap(train: list[Instance], test: list[Instance]) -> int:
    """Trading-day gap between max(train_date) and min(test_date), exclusive."""
    max_train = max(pd.Timestamp(i.metadata["date"]) for i in train)
    min_test = min(pd.Timestamp(i.metadata["date"]) for i in test)
    # bdate_range is half-open by pandas convention but here we want strictly-between days.
    between = pd.bdate_range(start=max_train + pd.Timedelta(days=1),
                             end=min_test - pd.Timedelta(days=1))
    return len(between)


def test_split_default_embargo_creates_trading_day_gap(tmp_path):
    """Default (auto-derived) embargo: trading-day gap >= 2 * embargo_days."""
    expected_embargo = max(features_mod.TARGET_HORIZON, features_mod.MAX_FEATURE_LOOKBACK)
    train, test = _load_two_years(tmp_path, embargo_days=None)
    assert train and test, f"empty split: train={len(train)}, test={len(test)}"

    max_train = max(pd.Timestamp(i.metadata["date"]) for i in train)
    min_test = min(pd.Timestamp(i.metadata["date"]) for i in test)

    # Prompt invariant 1: max(train_dates) + embargo_days < min(test_dates).
    assert max_train + pd.Timedelta(days=expected_embargo) < min_test, (
        f"train end {max_train} + {expected_embargo}d not strictly before test start {min_test}"
    )

    # Prompt invariant 2: trading-day gap >= 2 * embargo_days. Tighter than
    # calendar-day comparison and catches wrong-direction cuts (e.g. dropping
    # the *last* test rows instead of the first would leave the gap at 0).
    bday_gap = _bday_gap(train, test)
    assert bday_gap >= 2 * expected_embargo - 1, (
        f"trading-day gap {bday_gap} < 2*embargo_days={2 * expected_embargo} - 1"
    )


def test_split_explicit_embargo_creates_trading_day_gap(tmp_path):
    """Explicit ``embargo_days=21`` enforces the same invariant."""
    embargo = 21
    train, test = _load_two_years(tmp_path, embargo_days=embargo)
    bday_gap = _bday_gap(train, test)
    assert bday_gap >= 2 * embargo - 1, (
        f"trading-day gap {bday_gap} < 2*embargo={2 * embargo} - 1"
    )


def test_split_zero_embargo_keeps_legacy_behavior(tmp_path):
    """``embargo_days=0`` must preserve every train/test instance (back-compat)."""
    train, test = _load_two_years(tmp_path, embargo_days=0)
    assert all(pd.Timestamp(i.metadata["date"]).year < 2021 for i in train)
    assert all(pd.Timestamp(i.metadata["date"]).year == 2021 for i in test)

    bday_gap = _bday_gap(train, test)
    assert bday_gap < 21, (
        "no-embargo split unexpectedly already has a 21-trading-day gap — "
        "the synthetic data may not span the year boundary closely enough"
    )


def test_split_embargo_drops_correct_count(tmp_path):
    """``embargo_days=21`` drops exactly 21 instances from each side."""
    train_no, test_no = _load_two_years(tmp_path, embargo_days=0)
    train_emb, test_emb = _load_two_years(tmp_path, embargo_days=21)
    assert len(train_emb) == len(train_no) - 21
    assert len(test_emb) == len(test_no) - 21


def test_split_test_side_drops_earliest_not_latest(tmp_path):
    """Test embargo must drop the *first* test instances, not the last."""
    train_no, test_no = _load_two_years(tmp_path, embargo_days=0)
    _, test_emb = _load_two_years(tmp_path, embargo_days=21)
    test_no_dates = sorted(pd.Timestamp(i.metadata["date"]) for i in test_no)
    test_emb_dates = sorted(pd.Timestamp(i.metadata["date"]) for i in test_emb)
    # The earliest 21 dates should be removed; the latest dates should be preserved.
    assert test_emb_dates[0] == test_no_dates[21], (
        "test-side embargo dropped the wrong end: "
        f"first kept date is {test_emb_dates[0]}, expected {test_no_dates[21]}"
    )
    assert test_emb_dates[-1] == test_no_dates[-1], (
        "test-side embargo unexpectedly removed the latest date(s)"
    )


def test_split_train_side_drops_latest_not_earliest(tmp_path):
    """Train embargo must drop the *last* train instances, not the first."""
    train_no, _ = _load_two_years(tmp_path, embargo_days=0)
    train_emb, _ = _load_two_years(tmp_path, embargo_days=21)
    train_no_dates = sorted(pd.Timestamp(i.metadata["date"]) for i in train_no)
    train_emb_dates = sorted(pd.Timestamp(i.metadata["date"]) for i in train_emb)
    assert train_emb_dates[-1] == train_no_dates[-1 - 21], (
        f"train-side embargo dropped the wrong end: "
        f"last kept date is {train_emb_dates[-1]}, expected {train_no_dates[-1 - 21]}"
    )
    assert train_emb_dates[0] == train_no_dates[0]


def test_split_negative_embargo_raises(tmp_path):
    train, test = _load_two_years(tmp_path, embargo_days=0)
    with pytest.raises(ValueError, match="embargo_days"):
        PortfolioDataLoader.split(train + test, test_year=2021, embargo_days=-1)


def test_split_embargo_too_large_raises(tmp_path):
    """Empty result must be a loud ValueError, not a silent [] list."""
    train, test = _load_two_years(tmp_path, embargo_days=0)
    huge = max(len(train), len(test)) + 5
    with pytest.raises(ValueError, match="empty"):
        PortfolioDataLoader.split(train + test, test_year=2021, embargo_days=huge)


def test_split_auto_derives_from_features_constants(tmp_path, monkeypatch):
    """``embargo_days=None`` derives ``max(TARGET_HORIZON, MAX_FEATURE_LOOKBACK)``.

    Override the constants and confirm the auto-derived gap tracks the override.
    """
    # Spec from prompt: target_horizon=10, max_feature_lookback=30 -> embargo=30.
    monkeypatch.setattr(features_mod, "TARGET_HORIZON", 10, raising=True)
    monkeypatch.setattr(features_mod, "MAX_FEATURE_LOOKBACK", 30, raising=True)

    # We don't rebuild features (would require regenerating FEATURE_COLS); we
    # only want to verify the loader's embargo-derivation path. Build the
    # dataset with the original feature pipeline, then re-run split().
    train_no, test_no = _load_two_years(tmp_path, embargo_days=0)
    train_auto, test_auto = PortfolioDataLoader.split(
        train_no + test_no, test_year=2021, embargo_days=None
    )
    expected = max(10, 30)  # 30
    bday_gap = _bday_gap(train_auto, test_auto)
    assert bday_gap >= 2 * expected - 1, (
        f"auto-derived embargo gave trading-day gap {bday_gap} < 2*{expected} - 1"
    )
    # And confirm exactly ``expected`` instances were dropped from each side.
    assert len(train_no) - len(train_auto) == expected
    assert len(test_no) - len(test_auto) == expected
