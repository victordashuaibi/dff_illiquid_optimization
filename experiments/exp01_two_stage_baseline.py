"""Two-stage baseline: yfinance -> XGBoost (MSE) -> static Markowitz -> NDR.

Mini-config defaults (5 large caps, 2020-2022, test_year=2022) so the whole
pipeline runs end-to-end in a few minutes. Override via CLI flags or by
editing ``configs/default.yaml``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.backbone.xgb import XGBoostBackbone
from src.data.loader import Instance, PortfolioDataLoader
from src.optimizer.markowitz_static import MarkowitzStatic
from src.utils.seed import set_seed

# NB: torch is intentionally NOT imported here. In this venv, importing torch
# (built against numpy 1.x) alongside xgboost (also numpy-1.x ABI) under
# numpy 2.x segfaults during xgb.fit(). NDR is computed with numpy below;
# the unit tests still exercise the torch-based regret loss separately.


# Mini-experiment XGBoost defaults: the legacy n_estimators=800 was tuned on
# 263 tickers x 6 years (~380k panel rows). On a 5-ticker x 3-year mini run
# (~2k rows) those settings overfit aggressively, so we shrink to a lighter
# config. Re-introduce the legacy config when n_assets >= 30 or n_train >> 10k.
XGB_KWARGS_MINI: dict = dict(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+",
                   default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2022-12-31")
    p.add_argument("--test-year", type=int, default=2022)
    p.add_argument("--cov-window", type=int, default=60)
    p.add_argument("--risk-aversion", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/exp01")
    p.add_argument("--n-estimators", type=int, default=None,
                   help="Override mini-config n_estimators. Default uses XGB_KWARGS_MINI.")
    return p.parse_args()


def stack_w(ws: list[np.ndarray]) -> np.ndarray:
    return np.stack(ws, axis=0) if ws else np.empty((0, 0))


def stack_c(insts: list[Instance]) -> np.ndarray:
    return np.stack([i.c_true for i in insts], axis=0) if insts else np.empty((0, 0))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp01] tickers={args.tickers}")
    print(f"[exp01] window={args.start_date}..{args.end_date}, test_year={args.test_year}")

    # 1) Data
    loader = PortfolioDataLoader(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cov_window=args.cov_window,
    )
    instances = loader.load()
    print(f"[exp01] total instances: {len(instances)}")
    train, test = loader.split(instances, test_year=args.test_year)
    if not train or not test:
        raise RuntimeError(f"Empty split: train={len(train)}, test={len(test)}")
    print(f"[exp01] train={len(train)}, test={len(test)}")

    n_assets = train[0].X.shape[0]

    # 2) Backbone (panel-style global XGBoost on MSE)
    xgb_kwargs = dict(XGB_KWARGS_MINI)
    if args.n_estimators is not None:
        xgb_kwargs["n_estimators"] = args.n_estimators
    print(f"[exp01] xgb_kwargs={xgb_kwargs}")
    backbone = XGBoostBackbone(**xgb_kwargs)
    backbone.fit(train)
    c_hat_train = backbone.predict(train)
    c_hat = backbone.predict(test)
    c_true_train = stack_c(train)
    c_true = stack_c(test)
    mse_train = float(np.mean((c_hat_train - c_true_train) ** 2))
    mse_test = float(np.mean((c_hat - c_true) ** 2))
    target_std = float(c_true.std())
    rmse_test = float(np.sqrt(mse_test))
    # Baseline-MSE = predicting the train target mean for every test point.
    train_mean = float(c_true_train.mean())
    mse_naive = float(np.mean((c_true - train_mean) ** 2))
    print(f"[exp01] backbone MSE train: {mse_train:.6e}")
    print(f"[exp01] backbone MSE test : {mse_test:.6e}  "
          f"(naive train-mean MSE: {mse_naive:.6e})")
    print(f"[exp01] target std (test) : {target_std:.6e}")
    print(f"[exp01] RMSE test         : {rmse_test:.6e}")

    # 3) Static Markowitz on predicted vs. true returns
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

    # 4) Two NDR variants — both computed with numpy because importing torch
    # in the same process as xgboost segfaults under numpy 2.x in this venv;
    # the torch-based regret functions are still exercised in test_baseline.py.
    #
    # (i) Linear NDR (legacy / SPO-style; expected to be wrong-signed for
    #     Markowitz — surfaced as a sanity check, not the primary metric).
    obj_pred_lin = (c_true * w_pred_arr).sum(axis=-1)
    obj_oracle_lin = (c_true * w_oracle_arr).sum(axis=-1)
    ndr_linear = float(
        (obj_pred_lin - obj_oracle_lin).sum()
        / max(np.abs(obj_oracle_lin).sum(), 1e-8)
    )

    # (ii) Markowitz NDR (primary metric):
    #     f(w, c, Σ) = -c^T w + γ w^T Σ w  (minimization form)
    Sigmas = np.stack([inst.Sigma for inst in test], axis=0)  # [B, n, n]
    quad_pred = np.einsum("bi,bij,bj->b", w_pred_arr, Sigmas, w_pred_arr)
    quad_oracle = np.einsum("bi,bij,bj->b", w_oracle_arr, Sigmas, w_oracle_arr)
    f_pred = -obj_pred_lin + args.risk_aversion * quad_pred
    f_oracle = -obj_oracle_lin + args.risk_aversion * quad_oracle
    regret_per_inst = f_pred - f_oracle  # >= 0 up to solver tol
    min_regret = float(regret_per_inst.min())
    ndr_markowitz = float(
        regret_per_inst.sum() / max(np.abs(f_oracle).sum(), 1e-8)
    )

    obj_pred = float(obj_pred_lin.sum())
    obj_oracle = float(obj_oracle_lin.sum())

    # 5) Diagnostics
    # Sanity: do w_pred and w_oracle actually differ? If they collapse to
    # the same point (e.g. both at uniform), the backbone hasn't learned
    # anything actionable.
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
    # cvxpy ECOS tolerance ~1e-8 per solve, but with Cholesky + reweighting
    # the per-instance error can drift to ~1e-5; flag only larger negatives.
    print(f"  min per-instance Markowitz regret: {min_regret:.3e} "
          f"({'OK (solver tol)' if min_regret >= -1e-4 else 'NEGATIVE — bug'})")
    print(f"  NDR (linear, legacy) : {ndr_linear:.6f}  ({ndr_linear * 100:.3f}%)")
    print(f"  NDR (Markowitz)      : {ndr_markowitz:.6f}  ({ndr_markowitz * 100:.3f}%)")
    print("=" * 64)

    # 6) Save artefacts
    metrics = {
        "tickers": list(args.tickers),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "test_year": args.test_year,
        "n_train": len(train),
        "n_test": len(test),
        "n_assets": n_assets,
        "xgb_kwargs": xgb_kwargs,
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
    axes[1].set_xlabel("c_true"); axes[1].set_ylabel("c_hat")
    axes[1].set_title("Calibration scatter")
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=120)
    print(f"[exp01] saved metrics + predictions to {output_dir}/")


if __name__ == "__main__":
    main()
