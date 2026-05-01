"""Tests for the Two-stage baseline pipeline (universe / features / loader / backbone / static optimizer).

These tests use synthetic data only — no yfinance downloads — by either
monkey-patching ``PortfolioDataLoader._download_prices`` or skipping the
download path entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backbone.xgb import XGBoostBackbone
from src.data.features import FEATURE_COLS, TARGET_COL, make_features
from src.data.loader import Instance, PortfolioDataLoader
from src.data.universe import screen_by_illiquidity
from src.optimizer.markowitz_static import MarkowitzStatic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synth_prices(
    tickers: list[str],
    n_days: int = 200,
    start: str = "2020-01-02",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a long-format OHLCV DataFrame with random-walk Adj Close."""
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


# ---------------------------------------------------------------------------
# Step 2: universe screening
# ---------------------------------------------------------------------------
def test_universe_screening_returns_top_pct():
    rng = np.random.default_rng(0)
    n_tickers = 20
    n_days = 100
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    price_data: dict[str, pd.DataFrame] = {}
    for k in range(n_tickers):
        # Volume scaled by k -> high-k is more liquid; low-k is illiquid.
        vol_scale = 10 ** (3 + k * 0.2)
        px = 50.0 + np.cumsum(rng.normal(0, 0.5, n_days))
        price_data[f"T{k:02d}"] = pd.DataFrame({
            "Date": dates,
            "Close": px, "Adj Close": px,
            "High": px + 1.0, "Low": px - 1.0,
            "Volume": (vol_scale * (1 + rng.uniform(0, 0.1, n_days))).astype(int),
        })

    kept = screen_by_illiquidity(price_data, top_pct=0.25, require_full_history=False)
    assert len(kept) == int(np.ceil(20 * 0.25)) == 5
    # Lowest-volume tickers should dominate the kept set.
    assert "T00" in kept


def test_universe_screening_empty():
    assert screen_by_illiquidity({}, top_pct=0.25) == []


# ---------------------------------------------------------------------------
# Step 3: feature engineering — no look-ahead
# ---------------------------------------------------------------------------
def test_features_no_lookahead_on_monotone_series():
    """If price strictly increases, features at row t must depend only on t-1 and earlier."""
    n = 60
    dates = pd.bdate_range("2020-01-02", periods=n)
    px = np.linspace(100.0, 160.0, n)  # strictly increasing
    df = pd.DataFrame({
        "Date": dates, "Ticker": "X",
        "Open": px, "High": px + 0.5, "Low": px - 0.5,
        "Close": px, "Adj Close": px,
        "Volume": np.full(n, 1_000_000, dtype=int),
    })
    feats = make_features(df).set_index("Date").sort_index()

    # For a strictly increasing series, ret_lag1 at row t equals
    # the day-(t-1) return — must NOT include the day-t price jump.
    ret_lag1 = feats["ret_lag1"]
    raw_returns = pd.Series(px, index=dates).pct_change()
    expected = raw_returns.shift(1)
    aligned = pd.concat([ret_lag1, expected], axis=1).dropna()
    assert np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], atol=1e-12)

    # log_price uses today's price, but every other engineered feature uses
    # .shift(1)+rolling, so we sanity-check that ret_mean5 at t equals
    # mean(returns[t-5..t-1]) — i.e., excludes today.
    rm5 = feats["ret_mean5"].dropna()
    for date in rm5.index[:5]:
        loc = dates.get_loc(date)
        # Window is days [loc-5..loc-1] of the *return* series.
        window = raw_returns.iloc[loc - 5:loc]
        if window.isna().any():
            continue
        assert rm5.loc[date] == pytest.approx(window.mean(), abs=1e-12)


# ---------------------------------------------------------------------------
# Step 4: PortfolioDataLoader instance assembly
# ---------------------------------------------------------------------------
def test_dataloader_instance_shapes(tmp_path):
    tickers = ["A", "B", "C"]
    prices = _synth_prices(tickers, n_days=200, seed=123)

    loader = PortfolioDataLoader(
        tickers=tickers, start_date="2020-01-01", end_date="2021-01-01",
        cov_window=30, cache_dir=tmp_path, use_cache=False,
    )
    # Bypass yfinance entirely.
    loader._download_prices = lambda: prices  # type: ignore[assignment]

    instances = loader.load()
    assert len(instances) > 0

    n_assets = len(tickers)
    n_features = len(FEATURE_COLS)
    for k, inst in enumerate(instances):
        assert inst.X.shape == (n_assets, n_features), f"instance {k} X shape"
        assert inst.c_true.shape == (n_assets,), f"instance {k} c_true shape"
        assert inst.Sigma.shape == (n_assets, n_assets), f"instance {k} Sigma shape"
        assert inst.metadata["ticker_list"] == sorted(tickers)
        assert not np.isnan(inst.X).any()
        assert not np.isnan(inst.c_true).any()
        # PSD + symmetric.
        assert np.allclose(inst.Sigma, inst.Sigma.T, atol=1e-10)
        assert np.linalg.eigvalsh(inst.Sigma).min() >= -1e-8


def test_dataloader_split_by_year(tmp_path):
    tickers = ["A", "B"]
    prices = _synth_prices(tickers, n_days=400, start="2020-01-02", seed=7)

    loader = PortfolioDataLoader(
        tickers=tickers, start_date="2020-01-02", end_date="2022-01-01",
        cov_window=20, cache_dir=tmp_path, use_cache=False,
    )
    loader._download_prices = lambda: prices  # type: ignore[assignment]
    instances = loader.load()
    train, test = loader.split(instances, test_year=2021)
    # Every train instance is < 2021, every test instance is == 2021.
    assert all(i.metadata["date"].year < 2021 for i in train)
    assert all(i.metadata["date"].year == 2021 for i in test)
    assert len(train) + len(test) <= len(instances)


# ---------------------------------------------------------------------------
# Step 5: XGBoostBackbone
# ---------------------------------------------------------------------------
def test_backbone_fit_predict_shapes():
    rng = np.random.default_rng(0)
    n_assets = 4
    n_features = 8

    def mk(n: int) -> list[Instance]:
        out = []
        for t in range(n):
            X = rng.normal(size=(n_assets, n_features))
            c = X[:, 0] * 0.05 + rng.normal(0, 0.01, n_assets)
            Sigma = np.eye(n_assets) * 0.01
            out.append(Instance(X=X, c_true=c, Sigma=Sigma,
                                metadata={"date": t, "ticker_list": list("ABCD")}))
        return out

    train, test = mk(40), mk(10)
    bb = XGBoostBackbone(n_estimators=20)
    bb.fit(train)
    preds = bb.predict(test)
    assert preds.shape == (10, n_assets)
    assert np.isfinite(preds).all()


def test_backbone_predict_before_fit_raises():
    bb = XGBoostBackbone()
    with pytest.raises(RuntimeError, match="before fit"):
        bb.predict([])


# ---------------------------------------------------------------------------
# Step 6: MarkowitzStatic
# ---------------------------------------------------------------------------
def test_markowitz_static_constraints():
    rng = np.random.default_rng(0)
    n = 6
    opt = MarkowitzStatic(n_assets=n, risk_aversion=1.0, long_only=True)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T + 0.01 * np.eye(n)  # PSD
    c = rng.normal(size=n) * 0.05
    w = opt.solve(c, Sigma)
    assert w.shape == (n,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-8).all()


def test_markowitz_static_against_ground_truth_n2():
    """n=2 hand-computed: c=[0.05, 0.02], Sigma=0.04*I, gamma=1, long-only.

    Lagrangian gives w0 = 0.6875, w1 = 0.3125 (interior optimum).
    """
    opt = MarkowitzStatic(n_assets=2, risk_aversion=1.0, long_only=True)
    w = opt.solve(np.array([0.05, 0.02]), np.eye(2) * 0.04)
    assert np.allclose(w, [0.6875, 0.3125], atol=1e-4)


def test_markowitz_static_batch():
    opt = MarkowitzStatic(n_assets=3, risk_aversion=1.0, long_only=True)
    c_batch = np.array([
        [0.05, 0.02, 0.0],
        [0.0, 0.03, 0.04],
        [0.10, 0.0, 0.0],
    ])
    Sigma = np.eye(3) * 0.04
    w_batch = opt.solve_batch(c_batch, Sigma)
    assert w_batch.shape == (3, 3)
    assert np.allclose(w_batch.sum(axis=1), 1.0, atol=1e-6)
    assert (w_batch >= -1e-8).all()


# ---------------------------------------------------------------------------
# Step A: Markowitz decision regret / NDR
# ---------------------------------------------------------------------------
# This venv ships torch built against numpy 1.x but has numpy 2.0 installed,
# so torch.from_numpy / torch.tensor(np_array) raise "Numpy is not available".
# We convert via .tolist() which goes through Python lists and avoids the
# broken bridge. Production code never relies on this.
def _to_torch(arr):
    import torch
    return torch.tensor(arr.tolist(), dtype=torch.float64)


def test_markowitz_regret_zero_at_oracle():
    """When w_pred == w_oracle, regret must be zero per instance."""
    import torch
    from src.losses.regret import (
        markowitz_decision_regret,
        markowitz_normalized_decision_regret,
    )

    rng = np.random.default_rng(42)
    n_assets, B = 4, 6
    c_np = rng.normal(size=(B, n_assets)) * 0.05
    A = rng.normal(size=(n_assets, n_assets))
    Sigma_np = A @ A.T + 0.01 * np.eye(n_assets)

    opt = MarkowitzStatic(n_assets=n_assets, risk_aversion=1.0, long_only=True)
    w_oracle_np = np.stack([opt.solve(c_np[i], Sigma_np) for i in range(B)])

    c = _to_torch(c_np)
    Sigma = _to_torch(Sigma_np)
    w_oracle = _to_torch(w_oracle_np)

    r = markowitz_decision_regret(c, w_oracle, w_oracle, Sigma, risk_aversion=1.0)
    assert r.shape == (B,)
    assert torch.allclose(r, torch.zeros(B, dtype=torch.float64), atol=1e-6)
    ndr = markowitz_normalized_decision_regret(c, w_oracle, w_oracle, Sigma, risk_aversion=1.0)
    assert ndr.shape == torch.Size([])
    assert ndr.item() == pytest.approx(0.0, abs=1e-6)


def test_markowitz_regret_nonnegative():
    """Random feasible w_pred must give regret >= 0 vs the true oracle (within tol)."""
    from src.losses.regret import markowitz_decision_regret

    rng = np.random.default_rng(0)
    n_assets, B = 5, 10
    c_np = rng.normal(size=(B, n_assets)) * 0.05
    A = rng.normal(size=(n_assets, n_assets))
    Sigma_np = A @ A.T + 0.05 * np.eye(n_assets)

    opt = MarkowitzStatic(n_assets=n_assets, risk_aversion=1.0, long_only=True)
    w_oracle_np = np.stack([opt.solve(c_np[i], Sigma_np) for i in range(B)])
    # Random feasible w_pred (uniform on the simplex).
    w_pred_np = rng.dirichlet(np.ones(n_assets), size=B)

    r = markowitz_decision_regret(
        _to_torch(c_np),
        _to_torch(w_pred_np),
        _to_torch(w_oracle_np),
        _to_torch(Sigma_np),
        risk_aversion=1.0,
    )
    assert (r >= -1e-6).all(), f"min regret {float(r.min())}"


def test_markowitz_ndr_against_manual_calc_n2():
    """Hand-computed NDR for n_assets=2, c=[0.05, 0.02], Sigma=0.04*I, gamma=1.

    Oracle w = [0.6875, 0.3125] (interior); pick w_pred=[0.5, 0.5].
        f_oracle = -(0.05*0.6875 + 0.02*0.3125) + 0.04*(0.6875^2 + 0.3125^2)
                 = -0.040625 + 0.04 * 0.5703125 = -0.0178125
        f_pred   = -(0.05*0.5 + 0.02*0.5) + 0.04*(0.25 + 0.25)
                 = -0.035 + 0.02 = -0.015
        regret   = f_pred - f_oracle = 0.0028125
        NDR      = regret / |f_oracle| = 0.0028125 / 0.0178125 ≈ 0.157895
    """
    import torch
    from src.losses.regret import (
        markowitz_decision_regret,
        markowitz_normalized_decision_regret,
    )

    c = torch.tensor([[0.05, 0.02]], dtype=torch.float64)
    w_oracle = torch.tensor([[0.6875, 0.3125]], dtype=torch.float64)
    w_pred = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    Sigma = torch.tensor([[0.04, 0.0], [0.0, 0.04]], dtype=torch.float64)

    r = markowitz_decision_regret(c, w_pred, w_oracle, Sigma, risk_aversion=1.0)
    assert r.item() == pytest.approx(0.0028125, abs=1e-9)

    ndr = markowitz_normalized_decision_regret(c, w_pred, w_oracle, Sigma, risk_aversion=1.0)
    assert ndr.item() == pytest.approx(0.0028125 / 0.0178125, abs=1e-9)
