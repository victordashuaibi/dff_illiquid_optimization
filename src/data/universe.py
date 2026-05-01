"""Russell 2000 universe construction and ILLIQ-based screening.

Adapted from the legacy Two-stage pipeline (IWM holdings download +
composite ILLIQ/DollarVolume ranking). All hardcoded paths removed.
"""
from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests

IWM_HOLDINGS_URL = (
    "https://www.ishares.com/ch/professionals/en/products/239710/"
    "ishares-russell-2000-etf/1495092304805.ajax"
    "?dataType=fund&fileName=IWM_holdings&fileType=csv"
)


def get_russell_tickers(url: str = IWM_HOLDINGS_URL, timeout: int = 30) -> list[str]:
    """Fetch the current iShares Russell 2000 ETF (IWM) constituent list.

    The CSV served by iShares has a preamble; the holdings table starts at
    the first line beginning with ``Ticker,Name,``. Tickers like ``-`` (cash)
    are filtered out.
    """
    text = requests.get(url, timeout=timeout).text
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,Name,"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Cannot find IWM holdings table header in response.")

    holdings = pd.read_csv(StringIO("\n".join(lines[start_idx:])))
    tickers = (
        holdings["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    tickers = tickers[tickers != "-"]
    return tickers.drop_duplicates().tolist()


def screen_by_illiquidity(
    price_data: dict[str, pd.DataFrame],
    top_pct: float = 0.25,
    winsorize_bounds: tuple[float, float] = (0.01, 0.99),
    min_price: float = 5.0,
    require_full_history: bool = True,
) -> list[str]:
    """Composite ILLIQ + DollarVolume ranking; return the top ``top_pct`` tickers.

    Parameters
    ----------
    price_data : dict[ticker -> DataFrame]
        Each DataFrame must contain columns ``Date``, ``Close``, ``Adj Close``,
        ``Volume``. (Same schema yfinance produces with ``auto_adjust=False``.)
    top_pct : float
        Fraction of tickers to keep (e.g. 0.25 = top 25% most illiquid).
    winsorize_bounds : tuple[float, float]
        Per-ticker quantile bounds for ILLIQ winsorization.
    min_price : float
        Drop tickers whose median Adj Close is below this threshold.
    require_full_history : bool
        If True, drop tickers that don't span every trading day in the union.

    Returns
    -------
    list[str]
        Tickers ranked from most-illiquid to least, truncated at top_pct.

    Notes
    -----
    Replicates the legacy logic:
      - ILLIQ_t = |Return_t| / DollarVolume_t  (Amihud)
      - winsorize ILLIQ per-ticker at [q_low, q_high]
      - DollarVolume = Close * Volume
      - score = rank(avg_ILLIQ desc) + rank(avg_DollarVolume asc)
      - keep ceil(top_pct * N) lowest-score tickers (most illiquid).
    """
    if not price_data:
        return []

    frames = []
    for ticker, df in price_data.items():
        if df is None or df.empty:
            continue
        sub = df.copy()
        sub["Ticker"] = ticker
        frames.append(sub)
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ("Close", "Adj Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close", "Adj Close", "Volume"])
    df = df[(df["Adj Close"] > 0) & (df["Volume"] > 0)]
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    median_price = df.groupby("Ticker")["Adj Close"].median()
    keep = median_price[median_price >= min_price].index
    df = df[df["Ticker"].isin(keep)].copy()

    if require_full_history:
        n_target_dates = df["Date"].nunique()
        per_ticker_dates = df.groupby("Ticker")["Date"].nunique()
        complete = per_ticker_dates[per_ticker_dates == n_target_dates].index
        df = df[df["Ticker"].isin(complete)].copy()

    df["Return"] = df.groupby("Ticker")["Adj Close"].pct_change()
    df["DollarVolume"] = df["Close"] * df["Volume"]
    df["ILLIQ"] = df["Return"].abs() / df["DollarVolume"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Return", "DollarVolume", "ILLIQ"]).copy()

    q_low, q_high = winsorize_bounds
    lo = df.groupby("Ticker")["ILLIQ"].transform(lambda x: x.quantile(q_low))
    hi = df.groupby("Ticker")["ILLIQ"].transform(lambda x: x.quantile(q_high))
    df["ILLIQ"] = df["ILLIQ"].clip(lower=lo, upper=hi)

    liq = (
        df.groupby("Ticker")
        .agg(avg_ILLIQ=("ILLIQ", "mean"), avg_DollarVolume=("DollarVolume", "mean"))
        .reset_index()
    )
    liq["ILLIQ_rank"] = liq["avg_ILLIQ"].rank(ascending=False, method="average")
    liq["DVOL_rank"] = liq["avg_DollarVolume"].rank(ascending=True, method="average")
    liq["score"] = liq["ILLIQ_rank"] + liq["DVOL_rank"]
    liq = liq.sort_values("score", ascending=True).reset_index(drop=True)

    n_keep = int(np.ceil(len(liq) * top_pct))
    return liq.head(n_keep)["Ticker"].tolist()
