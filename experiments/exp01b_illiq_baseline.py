"""ILLIQ-screened Russell 2000 baseline (exp01b).

Pipeline mirrors exp01 but builds the universe from the *real* Russell 2000
ILLIQ ranking instead of hand-picked mega-caps:

    get_russell_tickers()          # ~1939 IWM holdings
    -> download OHLCV via yfinance  (cached on disk)
    -> screen_by_illiquidity(...)   # rank by composite ILLIQ + DollarVolume
    -> keep the top ``--n-keep`` (default 30) most illiquid names
    -> PortfolioDataLoader.load()   # build instances on the screened universe
    -> XGBoostBackbone (legacy hyperparams: n_estimators=800)
    -> MarkowitzStatic + Markowitz NDR

Usage
-----
First run on a small slice to validate the pipeline:
    PYTHONPATH=. python experiments/exp01b_illiq_baseline.py --max-universe-size 200

Then scale up to the full Russell:
    PYTHONPATH=. python experiments/exp01b_illiq_baseline.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backbone.xgb import XGBoostBackbone
from src.data.loader import Instance, PortfolioDataLoader
from src.data.universe import get_russell_tickers, screen_by_illiquidity
from src.optimizer.markowitz_static import MarkowitzStatic
from src.utils.seed import set_seed

# Legacy XGBoost hyperparams from stock_xgb_baseline.py (validated on the full
# 263-ticker Russell ILLIQ universe with ~380k panel rows). For mini configs
# (n_assets <= 10) use exp01 instead.
XGB_KWARGS_LEGACY: dict = dict(
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--end-date", default="2023-12-31")
    p.add_argument("--test-year", type=int, default=2023)
    p.add_argument("--cov-window", type=int, default=60)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-keep", type=int, default=30,
                   help="How many top-ILLIQ tickers to keep.")
    p.add_argument("--max-universe-size", type=int, default=None,
                   help="Cap the Russell holdings list (deterministic prefix). "
                        "Use ~200 for a quick pipeline smoke test before scaling up.")
    p.add_argument("--output-dir", default="results/exp01b")
    p.add_argument("--cache-dir", default="data/processed")
    return p.parse_args()


def stack_w(ws: list[np.ndarray]) -> np.ndarray:
    return np.stack(ws, axis=0) if ws else np.empty((0, 0))


def stack_c(insts: list[Instance]) -> np.ndarray:
    return np.stack([i.c_true for i in insts], axis=0) if insts else np.empty((0, 0))


def download_full_universe(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str,
) -> pd.DataFrame:
    """Use the loader's cached download path; returns long-format prices."""
    full_loader = PortfolioDataLoader(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cov_window=1,  # unused for download
        cache_dir=cache_dir,
    )
    return full_loader._download_prices()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Universe construction
    # ------------------------------------------------------------------ #
    print("[exp01b] fetching IWM holdings ...")
    t0 = time.time()
    russell_tickers = get_russell_tickers()
    print(f"[exp01b] Russell holdings: {len(russell_tickers)} tickers "
          f"(took {time.time() - t0:.1f}s)")
    if args.max_universe_size is not None:
        russell_tickers = russell_tickers[: args.max_universe_size]
        print(f"[exp01b] capped to first {len(russell_tickers)} tickers "
              f"(--max-universe-size)")

    print(f"[exp01b] downloading prices for {len(russell_tickers)} tickers "
          f"({args.start_date}..{args.end_date}) — cache hits if rerun")
    t0 = time.time()
    prices_long = download_full_universe(
        russell_tickers, args.start_date, args.end_date, args.cache_dir
    )
    n_present = prices_long["Ticker"].nunique()
    print(f"[exp01b] downloaded prices for {n_present}/{len(russell_tickers)} "
          f"tickers (took {time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 2) ILLIQ screening
    # ------------------------------------------------------------------ #
    price_dict = {tk: g.copy() for tk, g in prices_long.groupby("Ticker", sort=False)}
    print(f"[exp01b] screening by ILLIQ + DollarVolume composite rank ...")
    t0 = time.time()
    # top_pct=1.0 returns ALL surviving tickers sorted by illiquidity; slice n_keep below.
    sorted_tickers = screen_by_illiquidity(
        price_dict,
        top_pct=1.0,
        winsorize_bounds=(0.01, 0.99),
        min_price=5.0,
        require_full_history=True,
    )
    print(f"[exp01b] {len(sorted_tickers)} tickers passed full-history + "
          f"price>=$5 filter (took {time.time() - t0:.1f}s)")
    if len(sorted_tickers) < args.n_keep:
        raise RuntimeError(
            f"Only {len(sorted_tickers)} tickers survived screening, need >= {args.n_keep}"
        )
    top_tickers = sorted_tickers[: args.n_keep]
    print(f"[exp01b] kept top {len(top_tickers)} most illiquid tickers")
    print(f"[exp01b] symbols: {top_tickers}")

    # ------------------------------------------------------------------ #
    # 3) Build instances on the screened universe
    # ------------------------------------------------------------------ #
    loader = PortfolioDataLoader(
        tickers=top_tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cov_window=args.cov_window,
        cache_dir=args.cache_dir,
    )
    # Inject the already-downloaded prices to avoid a second yfinance round-trip.
    sub_prices = prices_long[prices_long["Ticker"].isin(top_tickers)].reset_index(drop=True)
    loader._download_prices = lambda: sub_prices  # type: ignore[assignment]
    instances = loader.load()
    print(f"[exp01b] total instances: {len(instances)}")
    train, test = loader.split(instances, test_year=args.test_year)
    if not train or not test:
        raise RuntimeError(f"Empty split: train={len(train)}, test={len(test)}")
    print(f"[exp01b] train={len(train)}, test={len(test)}")

    n_assets = train[0].X.shape[0]

    # ------------------------------------------------------------------ #
    # 4) Backbone (panel-style global XGBoost — legacy hyperparams)
    # ------------------------------------------------------------------ #
    print(f"[exp01b] xgb_kwargs (legacy)={XGB_KWARGS_LEGACY}")
    backbone = XGBoostBackbone(**XGB_KWARGS_LEGACY)
    backbone.fit(train)
    c_hat_train = backbone.predict(train)
    c_hat = backbone.predict(test)
    c_true_train = stack_c(train)
    c_true = stack_c(test)
    mse_train = float(np.mean((c_hat_train - c_true_train) ** 2))
    mse_test = float(np.mean((c_hat - c_true) ** 2))
    target_std = float(c_true.std())
    rmse_test = float(np.sqrt(mse_test))
    train_mean = float(c_true_train.mean())
    mse_naive = float(np.mean((c_true - train_mean) ** 2))
    print(f"[exp01b] backbone MSE train: {mse_train:.6e}")
    print(f"[exp01b] backbone MSE test : {mse_test:.6e}  "
          f"(naive train-mean MSE: {mse_naive:.6e})")
    print(f"[exp01b] target std (test) : {target_std:.6e}")
    print(f"[exp01b] RMSE test         : {rmse_test:.6e}")

    # ------------------------------------------------------------------ #
    # 5) Markowitz allocation + NDR
    # ------------------------------------------------------------------ #
    optimizer = MarkowitzStatic(
        n_assets=n_assets,
        risk_aversion=args.risk_aversion,
        long_only=True,
    )
    w_pred, w_oracle = [], []
    for inst, c_hat_t in zip(test, c_hat):
        w_pred.append(optimizer.solve(c_hat_t, inst.Sigma))
        w_oracle.append(optimizer.solve(inst.c_true, inst.Sigma))
    w_pred_arr = stack_w(w_pred)
    w_oracle_arr = stack_w(w_oracle)

    # Linear NDR (legacy / wrong-signed for Markowitz, kept as sanity check).
    obj_pred_lin = (c_true * w_pred_arr).sum(axis=-1)
    obj_oracle_lin = (c_true * w_oracle_arr).sum(axis=-1)
    ndr_linear = float(
        (obj_pred_lin - obj_oracle_lin).sum()
        / max(np.abs(obj_oracle_lin).sum(), 1e-8)
    )
    # Markowitz NDR (primary metric).
    Sigmas = np.stack([inst.Sigma for inst in test], axis=0)
    quad_pred = np.einsum("bi,bij,bj->b", w_pred_arr, Sigmas, w_pred_arr)
    quad_oracle = np.einsum("bi,bij,bj->b", w_oracle_arr, Sigmas, w_oracle_arr)
    f_pred = -obj_pred_lin + args.risk_aversion * quad_pred
    f_oracle = -obj_oracle_lin + args.risk_aversion * quad_oracle
    regret_per_inst = f_pred - f_oracle
    min_regret = float(regret_per_inst.min())
    ndr_markowitz = float(
        regret_per_inst.sum() / max(np.abs(f_oracle).sum(), 1e-8)
    )

    obj_pred = float(obj_pred_lin.sum())
    obj_oracle = float(obj_oracle_lin.sum())

    # ------------------------------------------------------------------ #
    # 6) Diagnostics
    # ------------------------------------------------------------------ #
    w_diff_norm = float(np.linalg.norm(w_pred_arr - w_oracle_arr, axis=1).mean())
    pred_std_per_asset = c_hat.std(axis=0)
    print()
    print("=" * 64)
    print(f"  test instances     : {len(test)}")
    print(f"  n_assets           : {n_assets}")
    print(f"  MSE train / test   : {mse_train:.6e} / {mse_test:.6e}")
    print(f"  MSE naive (mean)   : {mse_naive:.6e}  "
          f"({'beats' if mse_test < mse_naive else 'WORSE THAN'} naive)")
    print(f"  RMSE test          : {rmse_test:.6e} (target std {target_std:.6e})")
    print(f"  c_hat std/asset    : "
          f"min={pred_std_per_asset.min():.3e} max={pred_std_per_asset.max():.3e}")
    print(f"  obj(c·w_pred sum)  : {obj_pred:.6e}")
    print(f"  obj(c·w_oracle sum): {obj_oracle:.6e}")
    print(f"  ||w_pred - w_oracle|| (avg L2): {w_diff_norm:.4f}")
    print(f"  min per-instance Markowitz regret: {min_regret:.3e} "
          f"({'OK (solver tol)' if min_regret >= -1e-4 else 'NEGATIVE — bug'})")
    print(f"  NDR (linear, legacy) : {ndr_linear:.6f}  ({ndr_linear * 100:.3f}%)")
    print(f"  NDR (Markowitz)      : {ndr_markowitz:.6f}  ({ndr_markowitz * 100:.3f}%)")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # 7) Save artefacts
    # ------------------------------------------------------------------ #
    metrics = {
        "tickers": list(top_tickers),
        "russell_holdings": len(russell_tickers),
        "n_present_after_download": int(n_present),
        "n_passed_screening": len(sorted_tickers),
        "n_keep": args.n_keep,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "test_year": args.test_year,
        "n_train": len(train),
        "n_test": len(test),
        "n_assets": n_assets,
        "xgb_kwargs": XGB_KWARGS_LEGACY,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "mse_naive": mse_naive,
        "rmse_test": rmse_test,
        "target_std": target_std,
        "obj_pred": obj_pred,
        "obj_oracle": obj_oracle,
        "min_per_instance_markowitz_regret": min_regret,
        "ndr_linear": ndr_linear,
        "ndr_markowitz": ndr_markowitz,
        "w_diff_avg_l2": w_diff_norm,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    np.savez(
        output_dir / "predictions.npz",
        c_hat=c_hat, c_true=c_true,
        w_pred=w_pred_arr, w_oracle=w_oracle_arr,
        dates=np.array([str(i.metadata["date"]) for i in test]),
        tickers=np.array(test[0].metadata["ticker_list"]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(c_hat.ravel(), bins=40, alpha=0.6, label="c_hat")
    axes[0].hist(c_true.ravel(), bins=40, alpha=0.6, label="c_true")
    axes[0].set_title("Test predictions vs. ground truth")
    axes[0].set_xlabel("21-day forward return")
    axes[0].legend()
    axes[1].scatter(c_true.ravel(), c_hat.ravel(), s=4, alpha=0.4)
    lo = float(min(c_true.min(), c_hat.min()))
    hi = float(max(c_true.max(), c_hat.max()))
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[1].set_xlabel("c_true")
    axes[1].set_ylabel("c_hat")
    axes[1].set_title("Calibration scatter")
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=120)
    print(f"[exp01b] saved metrics + predictions to {output_dir}/")


if __name__ == "__main__":
    main()
