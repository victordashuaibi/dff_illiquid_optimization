"""Panel-to-instance data loader for the Two-stage baseline.

Downloads OHLCV from yfinance (with on-disk parquet cache), computes
per-ticker features via :func:`src.data.features.make_features`, and
reshapes the result into a list of :class:`Instance` objects keyed by
trading day. Each :class:`Instance` packages the cross-sectional feature
matrix, the realised forward returns, and a Ledoit-Wolf shrinkage
covariance over the trailing ``cov_window`` days.

The instance format and invariants are defined in ``docs/INTERFACE.md``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data.features import FEATURE_COLS, TARGET_COL, make_features


@dataclass
class Instance:
    """One trading-day cross-section ready for the optimizer.

    Shapes follow ``docs/INTERFACE.md``:
      - X      : [n_assets, n_features_per_asset]
      - c_true : [n_assets]
      - Sigma  : [n_assets, n_assets]
    """
    X: np.ndarray
    c_true: np.ndarray
    Sigma: np.ndarray
    metadata: dict = field(default_factory=dict)


class PortfolioDataLoader:
    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        cov_window: int = 60,
        cache_dir: Optional[str | Path] = "data/processed",
        use_cache: bool = True,
    ):
        if not tickers:
            raise ValueError("tickers must be non-empty")
        # Deterministic ordering: alphabetical. The same ordering is reused
        # everywhere downstream so X[i, :], c_true[i], Sigma[i, *] all align.
        self.tickers: list[str] = sorted(set(tickers))
        self.start_date = start_date
        self.end_date = end_date
        self.cov_window = int(cov_window)
        self.use_cache = use_cache
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Raw price download (with parquet cache)
    # ------------------------------------------------------------------ #
    def _cache_key(self) -> str:
        s = "|".join(self.tickers) + f"|{self.start_date}|{self.end_date}"
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"prices_{self._cache_key()}.csv"

    def _download_prices(self) -> pd.DataFrame:
        """Return long-format OHLCV: ``Date, Ticker, Open, High, Low, Close, Adj Close, Volume``."""
        cache = self._cache_path()
        if self.use_cache and cache is not None and cache.exists():
            cached = pd.read_csv(cache)
            cached["Date"] = pd.to_datetime(cached["Date"])
            return cached

        import yfinance as yf

        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            raise RuntimeError(
                f"yfinance returned no data for {self.tickers} "
                f"in {self.start_date}..{self.end_date}"
            )

        # yfinance always returns MultiIndex columns (field, ticker), even
        # for a single ticker. Stack the ticker level to long format.
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, [self.tickers[0]]])
        long = raw.stack(level=-1, future_stack=True).reset_index()
        long = long.rename(columns={"level_1": "Ticker"})
        # Some yfinance versions name the stacked level 'Ticker' already; if not, find it.
        if "Ticker" not in long.columns:
            for c in long.columns:
                if c not in {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}:
                    long = long.rename(columns={c: "Ticker"})
                    break

        keep_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        long = long[[c for c in keep_cols if c in long.columns]].copy()
        long["Date"] = pd.to_datetime(long["Date"])
        long = long.dropna(subset=["Date", "Ticker"])
        long = long.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        if cache is not None:
            long.to_csv(cache, index=False)
        return long

    # ------------------------------------------------------------------ #
    # Instance assembly
    # ------------------------------------------------------------------ #
    def load(self) -> list[Instance]:
        prices = self._download_prices()
        # Restrict to tickers we actually got data for, preserving the
        # alphabetical ordering chosen in __init__.
        present = sorted(set(prices["Ticker"]).intersection(self.tickers))
        if len(present) != len(self.tickers):
            missing = sorted(set(self.tickers) - set(present))
            print(
                f"[PortfolioDataLoader] {len(missing)} tickers had no yfinance data "
                f"and will be dropped: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        self.tickers = present
        if not self.tickers:
            raise RuntimeError("No tickers had any yfinance data; cannot build instances.")
        prices = prices[prices["Ticker"].isin(self.tickers)].reset_index(drop=True)

        feats = make_features(prices)
        return self._build_instances(prices, feats)

    def _build_instances(
        self,
        prices: pd.DataFrame,
        feats: pd.DataFrame,
    ) -> list[Instance]:
        n_assets = len(self.tickers)
        n_features = len(FEATURE_COLS)

        # Wide return matrix [n_dates, n_assets] used for Ledoit-Wolf covariance.
        wide_ret = (
            prices.assign(_ret=prices.groupby("Ticker")["Adj Close"].pct_change())
            .pivot(index="Date", columns="Ticker", values="_ret")
            .reindex(columns=self.tickers)
            .sort_index()
        )

        # Wide feature tensor: dict[date] -> [n_assets, n_features]; built by
        # pivoting each feature column individually (cheaper than a 3D pivot).
        feats_idx = feats.set_index(["Date", "Ticker"]).sort_index()
        common_dates = sorted(set(wide_ret.index).intersection(
            feats_idx.index.get_level_values("Date").unique()
        ))

        # Pre-pivot each per-asset column once for O(n_dates) per-feature lookup.
        feat_panels: dict[str, pd.DataFrame] = {}
        for col in FEATURE_COLS:
            feat_panels[col] = (
                feats_idx[col]
                .unstack("Ticker")
                .reindex(columns=self.tickers)
                .sort_index()
            )
        target_panel = (
            feats_idx[TARGET_COL]
            .unstack("Ticker")
            .reindex(columns=self.tickers)
            .sort_index()
        )

        instances: list[Instance] = []
        lw = LedoitWolf()
        for t_idx, date in enumerate(common_dates):
            # --- features X[t] : [n_assets, n_features]
            X = np.empty((n_assets, n_features), dtype=float)
            any_nan_X = False
            for j, col in enumerate(FEATURE_COLS):
                if date not in feat_panels[col].index:
                    any_nan_X = True
                    break
                row = feat_panels[col].loc[date].to_numpy()
                if np.isnan(row).any():
                    any_nan_X = True
                    break
                X[:, j] = row
            if any_nan_X:
                continue

            # --- target c_true[t] : [n_assets]
            if date not in target_panel.index:
                continue
            c_true = target_panel.loc[date].to_numpy()
            if np.isnan(c_true).any():
                continue

            # --- covariance: trailing cov_window returns ending at date (exclusive).
            ret_loc = wide_ret.index.get_loc(date)
            if ret_loc < self.cov_window:
                continue
            ret_window = wide_ret.iloc[ret_loc - self.cov_window:ret_loc].to_numpy()
            if np.isnan(ret_window).any():
                continue
            try:
                Sigma = lw.fit(ret_window).covariance_
            except Exception:
                continue
            # Symmetrize defensively (Ledoit-Wolf already returns PSD).
            Sigma = 0.5 * (Sigma + Sigma.T)

            instances.append(Instance(
                X=X,
                c_true=c_true,
                Sigma=Sigma,
                metadata={
                    "date": pd.Timestamp(date),
                    "ticker_list": list(self.tickers),
                    "instance_id": t_idx,
                },
            ))

        self._sanity_check(instances)
        return instances

    # ------------------------------------------------------------------ #
    # Invariant enforcement (per docs/INTERFACE.md)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanity_check(instances: list[Instance]) -> None:
        if not instances:
            raise RuntimeError(
                "PortfolioDataLoader produced 0 instances. Likely causes: "
                "cov_window too long for date range, all dates have NaN, "
                "or yfinance returned partial data."
            )
        ref_tickers = instances[0].metadata["ticker_list"]
        n_assets = len(ref_tickers)
        for k, inst in enumerate(instances):
            tag = f"instance #{k} (date={inst.metadata.get('date')})"
            if inst.metadata["ticker_list"] != ref_tickers:
                raise AssertionError(f"{tag}: ticker_list differs from reference")
            if inst.X.shape != (n_assets, len(FEATURE_COLS)):
                raise AssertionError(f"{tag}: X shape {inst.X.shape} != ({n_assets},{len(FEATURE_COLS)})")
            if inst.c_true.shape != (n_assets,):
                raise AssertionError(f"{tag}: c_true shape {inst.c_true.shape} != ({n_assets},)")
            if inst.Sigma.shape != (n_assets, n_assets):
                raise AssertionError(f"{tag}: Sigma shape {inst.Sigma.shape} != ({n_assets},{n_assets})")
            if np.isnan(inst.X).any():
                bad = np.argwhere(np.isnan(inst.X))[0]
                raise AssertionError(f"{tag}: X has NaN at asset {bad[0]} feature {bad[1]}")
            if np.isnan(inst.c_true).any():
                bad = int(np.argwhere(np.isnan(inst.c_true))[0][0])
                raise AssertionError(f"{tag}: c_true has NaN at asset {bad} ({ref_tickers[bad]})")
            if np.isnan(inst.Sigma).any():
                raise AssertionError(f"{tag}: Sigma has NaN")
            if not np.allclose(inst.Sigma, inst.Sigma.T, atol=1e-8):
                raise AssertionError(f"{tag}: Sigma not symmetric")
            min_eig = float(np.linalg.eigvalsh(inst.Sigma).min())
            if min_eig < -1e-6:
                raise AssertionError(
                    f"{tag}: Sigma not PSD (min eigenvalue={min_eig:.2e})"
                )

    # ------------------------------------------------------------------ #
    # Train / test split (by calendar year)
    # ------------------------------------------------------------------ #
    @staticmethod
    def split(
        instances: list[Instance], test_year: int
    ) -> tuple[list[Instance], list[Instance]]:
        """Time-ordered split: train on dates with year < ``test_year``, test on the year itself."""
        train, test = [], []
        for inst in instances:
            year = pd.Timestamp(inst.metadata["date"]).year
            if year < test_year:
                train.append(inst)
            elif year == test_year:
                test.append(inst)
        return train, test
