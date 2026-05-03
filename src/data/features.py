"""Per-ticker feature engineering for the Two-stage baseline.

Generates 51 features + 1 forward-return target per (Date, Ticker) row.
All rolling/lag features use ``.shift(1)`` to prevent look-ahead bias.

Required input columns: ``Date``, ``Ticker``, ``Open``, ``High``, ``Low``,
``Close``, ``Adj Close``, ``Volume`` (matches yfinance ``auto_adjust=False``).
``Return``, ``DollarVolume`` and ``ILLIQ`` are computed if missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Forward-return horizon (trading days) used to build :data:`TARGET_COL`.
# The loader reads this when computing the train/test embargo so the gap
# tracks the actual leakage horizon — see ``PortfolioDataLoader.split``.
TARGET_HORIZON: int = 21

# Longest rolling/lag window used by ``_build_features_one_ticker`` (mirrors
# the ``rolling(20)`` calls and the ``shift(20)`` lag-return). The loader
# reads this for the embargo computation; bump it whenever a longer window
# is added to FEATURE_COLS.
MAX_FEATURE_LOOKBACK: int = 20

TARGET_COL = f"ret_fwd_{TARGET_HORIZON}d"

FEATURE_COLS: list[str] = (
    [f"ret_lag{lag}" for lag in range(1, MAX_FEATURE_LOOKBACK + 1)]
    + [
        "ret_mean5", "ret_mean10", "ret_mean20",
        "ret_vol5", "ret_vol10", "ret_vol20",
        "mom5", "mom10", "mom20",
        "daily_range", "range5", "range20", "price_pos20",
        "volume_mean5", "volume_mean20",
        "dollarvol_mean5", "dollarvol_mean20",
        "vol_ratio20", "dollarvol_ratio20",
        "illiq_lag1", "illiq_mean5", "illiq_mean20", "illiq_std20",
        "log_illiq_lag1", "log_illiq_mean20",
        "zero_return_count20", "zero_volume_count20",
        "abs_return_mean20",
        "log_price", "log_volume", "log_dollar_volume",
    ]
)


def _build_features_one_ticker(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-ticker features for a single sorted-by-Date frame."""
    g = g.sort_values("Date").copy()

    # --- Target: forward TARGET_HORIZON-day simple return (uses negative shift on purpose).
    g[TARGET_COL] = g["Adj Close"].shift(-TARGET_HORIZON) / g["Adj Close"] - 1

    # --- Lag returns 1..MAX_FEATURE_LOOKBACK.
    for lag in range(1, MAX_FEATURE_LOOKBACK + 1):
        g[f"ret_lag{lag}"] = g["Return"].shift(lag)

    # --- Rolling return statistics (mean / vol) over 5/10/20-day windows.
    for w in (5, 10, 20):
        g[f"ret_mean{w}"] = g["Return"].rolling(w).mean().shift(1)
        g[f"ret_vol{w}"] = g["Return"].rolling(w).std().shift(1)

    # --- Momentum (price ratios at 5/10/20-day horizons).
    g["mom5"] = g["Adj Close"].shift(1) / g["Adj Close"].shift(6) - 1
    g["mom10"] = g["Adj Close"].shift(1) / g["Adj Close"].shift(11) - 1
    g["mom20"] = g["Adj Close"].shift(1) / g["Adj Close"].shift(21) - 1

    # --- Intraday range and rolling range.
    g["daily_range"] = (g["High"] - g["Low"]) / g["Adj Close"]
    g["range5"] = g["daily_range"].rolling(5).mean().shift(1)
    g["range20"] = g["daily_range"].rolling(20).mean().shift(1)

    # --- Price position within 20-day high/low band.
    rolling_high20 = g["High"].rolling(20).max().shift(1)
    rolling_low20 = g["Low"].rolling(20).min().shift(1)
    g["price_pos20"] = (g["Adj Close"].shift(1) - rolling_low20) / (
        rolling_high20 - rolling_low20
    )

    # --- Volume / DollarVolume rolling means and ratios.
    g["volume_mean5"] = g["Volume"].rolling(5).mean().shift(1)
    g["volume_mean20"] = g["Volume"].rolling(20).mean().shift(1)
    g["dollarvol_mean5"] = g["DollarVolume"].rolling(5).mean().shift(1)
    g["dollarvol_mean20"] = g["DollarVolume"].rolling(20).mean().shift(1)
    g["vol_ratio20"] = g["Volume"].shift(1) / g["volume_mean20"]
    g["dollarvol_ratio20"] = g["DollarVolume"].shift(1) / g["dollarvol_mean20"]

    # --- ILLIQ (Amihud illiquidity) features.
    g["illiq_lag1"] = g["ILLIQ"].shift(1)
    g["illiq_mean5"] = g["ILLIQ"].rolling(5).mean().shift(1)
    g["illiq_mean20"] = g["ILLIQ"].rolling(20).mean().shift(1)
    g["illiq_std20"] = g["ILLIQ"].rolling(20).std().shift(1)
    g["log_illiq_lag1"] = np.log1p(g["illiq_lag1"])
    g["log_illiq_mean20"] = np.log1p(g["illiq_mean20"])

    # --- Zero-activity counts (proxy for trading frictions).
    zero_return = (g["Return"].abs() < 1e-12).astype(int)
    zero_volume = (g["Volume"] <= 0).astype(int)
    g["zero_return_count20"] = zero_return.rolling(20).sum().shift(1)
    g["zero_volume_count20"] = zero_volume.rolling(20).sum().shift(1)

    # --- Absolute return mean (volatility proxy that survives sign flips).
    g["abs_return_mean20"] = g["Return"].abs().rolling(20).mean().shift(1)

    # --- Log levels (price / volume / dollar volume).
    g["log_price"] = np.log(g["Adj Close"])
    g["log_volume"] = np.log1p(g["Volume"])
    g["log_dollar_volume"] = np.log1p(g["DollarVolume"])

    return g


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-ticker features for every (Date, Ticker) row in ``df``.

    Returns a DataFrame containing ``Date``, ``Ticker``, all 51 entries of
    :data:`FEATURE_COLS`, and the target :data:`TARGET_COL`. Rows where any
    feature is NaN (e.g. the first 20 days per ticker, or the last 21 due to
    forward-return computation) are *not* dropped here; downstream code is
    expected to handle masking.
    """
    required = {"Date", "Ticker", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"make_features missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if "Return" not in df.columns:
        df["Return"] = df.groupby("Ticker")["Adj Close"].pct_change()
    if "DollarVolume" not in df.columns:
        df["DollarVolume"] = df["Close"] * df["Volume"]
    if "ILLIQ" not in df.columns:
        df["ILLIQ"] = df["Return"].abs() / df["DollarVolume"]

    df["Volume"] = df["Volume"].clip(lower=0)
    df["DollarVolume"] = df["DollarVolume"].clip(lower=0)
    df["ILLIQ"] = df["ILLIQ"].clip(lower=0)
    df = df.replace([np.inf, -np.inf], np.nan)

    out = pd.concat(
        [_build_features_one_ticker(g) for _, g in df.groupby("Ticker", sort=False)],
        ignore_index=True,
    )
    out = out.replace([np.inf, -np.inf], np.nan)

    final_cols = ["Date", "Ticker"] + FEATURE_COLS + [TARGET_COL]
    return out[final_cols]
