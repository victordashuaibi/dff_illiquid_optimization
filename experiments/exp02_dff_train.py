"""DFF training pass on the Russell ILLIQ-30 universe (single config).

Pipeline (matches ``docs/exp02_design.md`` decisions D1–D7):

1. Load the deterministic ILLIQ-30 ticker list (frozen from exp01b's
   screening). Build instances, embargo-split into ``train`` / ``test``,
   then carve the last 10% of train (time-ordered) into ``val``. The
   remaining 90% is ``train_inner``.
2. **Cross-fit XGBRegressor on train_inner only** (val excluded) for
   leak-free OOF ``c_hat_train``. Discard fold models.
3. **Refit one M_full on train_inner** (val still excluded). Use M_full
   to produce ``c_hat_val`` and ``c_hat_test``.
4. **Static Sigma** (D2a): ``LedoitWolf().fit(c_true_train_inner)``.
   Same Σ used at oracle-cache build time and at training time.
5. Build oracle caches for train_inner / val / test via
   ``MarkowitzStatic.solve_batch``.
6. Standardize per-asset features with a ``StandardScaler`` fit on
   ``train_inner``; persist.
7. Build ``PerAssetBiasCorrectionLayer`` (D1b: shared per-asset NN with
   per-asset c_hat scalar concatenated to features).
8. Train F_θ with ``MarkowitzRegretLoss`` (D2a's static Σ) using Adam.
9. Per-epoch val: regret + Theorem 1 cosine and RMSE bounds asserted
   with full violator details (split, epoch, instance index, value, bound).
10. Final test eval: NDR(F_θ) vs NDR(M_full) two-stage baseline, both
    evaluated with the same ``Sigma_static`` so they are apples-to-apples.
11. Save artifacts under ``runs/exp02_dff_<UTC_timestamp>/``.

The core trainer logic lives in :func:`run_trainer`; ``main`` is a thin
wrapper that parses CLI flags, loads the universe, and invokes
``run_trainer`` with disk-side artifact saving. Tests
(``tests/test_determinism.py``) call ``run_trainer`` directly with
synthetic instances and ``run_dir=None`` to skip disk I/O.

Usage::

    PYTHONPATH=. python experiments/exp02_dff_train.py
    PYTHONPATH=. python experiments/exp02_dff_train.py --epsilon 0.2
"""
from __future__ import annotations

# OpenMP duplicate-runtime suppression — must precede any import that
# pulls in numpy/torch/xgboost. torch ships LLVM libomp; xgboost ships
# Intel libiomp; both loaded into one process deadlock during xgb.fit
# unless these env vars are set before the OpenMP runtimes initialize.
# See conftest.py for the full diagnosis (sample stack trace confirmed).
import os as _os  # noqa: E402
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# NB: xgboost MUST be imported before torch in this venv. torch loads
# libomp; xgboost loads libiomp; whichever gets initialized second
# segfaults on the first xgb.fit() call. The same pattern surfaces in
# experiments/exp01_two_stage_baseline.py (which sidesteps it by not
# importing torch at all). Confirmed minimal repro: torch-then-xgboost
# segfaults; xgboost-then-torch works.
import xgboost  # noqa: F401  — load OpenMP first
from xgboost import XGBRegressor

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

# Project imports.
from src.backbone.cross_fit import cross_fit_predict
from src.data.loader import Instance, PortfolioDataLoader
from src.dff.bias_correction import PerAssetBiasCorrectionLayer
from src.losses.markowitz_regret import (
    MarkowitzRegretLoss,
    build_oracle_cache,
)
from src.optimizer.markowitz_diff import DiffMarkowitz
from src.optimizer.markowitz_static import MarkowitzStatic
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Frozen universe (top-30 ILLIQ from exp01b's Russell screening, seed 42).
# Documented in docs/week01_findings.md and runs/exp01b/metrics.json.
# ---------------------------------------------------------------------------
ILLIQ_30_TICKERS: list[str] = [
    "LWAY", "USAU", "NODK", "TRAK", "FUNC", "MDWD", "GENC", "STRS",
    "ESCA", "RVSB", "TARA", "AIOT", "SAMG", "CIA",  "RGCO", "NATR",
    "GWRS", "ISTR", "UNTY", "VHI",  "GAIA", "SMHI", "MPX",  "GHM",
    "WEYS", "WNEB", "EVI",  "MFIN", "CLPR", "WLFC",
]


# ---------------------------------------------------------------------------
# Single-config dict (decision D5)
# ---------------------------------------------------------------------------
def _universe_hash(tickers: list[str]) -> str:
    universe_sorted = tuple(sorted(tickers))
    return hashlib.md5(repr(universe_sorted).encode()).hexdigest()[:8]


def base_config() -> dict:
    seed = 42
    universe_sorted = sorted(ILLIQ_30_TICKERS)
    return {
        # ---- universe identity ----
        "universe_tickers": list(universe_sorted),                  # full sorted list
        "universe_hash":    _universe_hash(ILLIQ_30_TICKERS),        # 8-char md5

        # ---- data ----
        "start_date":   "2018-01-01",
        "end_date":     "2023-12-31",
        "test_year":    2023,
        "cov_window":   60,                 # only used internally by loader; we use static Σ

        # ---- F_θ (paper §6.1) ----
        "hidden_dim":      32,
        "n_hidden_layers": 3,
        "epsilon":         0.5,             # paper synthetic default; sweep target

        # ---- optimizer (paper §6.1) ----
        "lr":         1e-3,
        "batch_size": 32,
        "epochs":     50,

        # ---- our additions ----
        "gamma":              1.0,          # match exp01b risk_aversion
        "embargo_days":       None,         # auto from features.TARGET_HORIZON / MAX_FEATURE_LOOKBACK
        "n_cross_fit_splits": 2,
        "val_fraction":       0.10,
        "seed":               seed,

        # ---- backbone (legacy from exp01b) ----
        "xgb_kwargs": {
            "objective":         "reg:squarederror",
            "n_estimators":      800,
            "learning_rate":     0.03,
            "max_depth":         4,
            "min_child_weight":  5,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "reg_alpha":         0.1,
            "reg_lambda":        2.0,
            "random_state":      seed,        # tracks config["seed"] for full determinism
            "n_jobs":            -1,
            "tree_method":       "hist",
        },

        # ---- runtime (filled in main) ----
        "cache_dir":  "data/processed",
        "output_dir": None,
    }


# ---------------------------------------------------------------------------
# Panel helpers — same flatten convention as XGBoostBackbone._stack_panel
# ---------------------------------------------------------------------------
def panel_X(instances: list[Instance]) -> np.ndarray:
    return np.concatenate([inst.X for inst in instances], axis=0)


def panel_c_true(instances: list[Instance]) -> np.ndarray:
    return np.stack([inst.c_true for inst in instances], axis=0)


def predict_panel(model: XGBRegressor, instances: list[Instance]) -> np.ndarray:
    """Apply ``model`` row-wise to each instance's X; return shape (N, n_assets)."""
    n_assets = instances[0].X.shape[0]
    out = np.empty((len(instances), n_assets), dtype=float)
    for i, inst in enumerate(instances):
        out[i] = model.predict(inst.X)
    return out


def time_ordered_val_split(
    train_all: list[Instance], val_fraction: float
) -> tuple[list[Instance], list[Instance]]:
    """Last ``val_fraction`` of train (sorted by date) becomes val (D3)."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    sorted_train = sorted(train_all, key=lambda i: pd.Timestamp(i.metadata["date"]))
    n_val = int(round(len(sorted_train) * val_fraction))
    if n_val == 0:
        raise ValueError(
            f"val_fraction={val_fraction} too small for {len(sorted_train)} train instances"
        )
    train_inner = sorted_train[: -n_val]
    val = sorted_train[-n_val:]
    return train_inner, val


def reshape_panel_to_per_asset(panel_x: np.ndarray, n_assets: int) -> np.ndarray:
    """(N*n_assets, n_features) → (N, n_assets, n_features)."""
    n_panel, n_features = panel_x.shape
    if n_panel % n_assets != 0:
        raise ValueError(
            f"panel rows {n_panel} not divisible by n_assets {n_assets}"
        )
    return panel_x.reshape(n_panel // n_assets, n_assets, n_features)


# ---------------------------------------------------------------------------
# Theorem 1 diagnostics + assertions — per-instance arrays preserved
# ---------------------------------------------------------------------------
@dataclass
class Theorem1Stats:
    cos_per_inst: torch.Tensor          # (B,)
    cos_lower_bound: float              # sqrt(1 - eps^2)
    rmse_delta_per_inst: torch.Tensor   # (B,)
    rmse_bound_per_inst: torch.Tensor   # (B,)


def theorem1_diagnostics(
    c_tilde: torch.Tensor, c_hat: torch.Tensor, c_true: torch.Tensor,
    epsilon: float, n_assets: int,
) -> Theorem1Stats:
    """Per-instance Theorem 1 quantities (Eq. 12 + Eq. 14)."""
    eps_sq = max(1.0 - epsilon * epsilon, 0.0)
    cos_lower = math.sqrt(eps_sq)
    cos_per = (c_tilde * c_hat).sum(dim=-1) / (
        c_tilde.norm(dim=-1) * c_hat.norm(dim=-1) + 1e-12
    )
    rmse_tilde = ((c_tilde - c_true) ** 2).mean(dim=-1).sqrt()
    rmse_hat = ((c_hat - c_true) ** 2).mean(dim=-1).sqrt()
    rmse_delta = rmse_tilde - rmse_hat
    rmse_bound = (epsilon / math.sqrt(n_assets)) * c_hat.norm(dim=-1)
    return Theorem1Stats(cos_per, cos_lower, rmse_delta, rmse_bound)


def assert_theorem1(
    stats: Theorem1Stats, *, label: str, epoch: Optional[int] = None,
    cos_slack: float = 1e-3, rmse_slack: float = 1e-3,
) -> None:
    """Raise ``RuntimeError`` with violator details if either Theorem 1 bound fires.

    Per-instance bounds (Eq. 12 + Eq. 14). Message includes split label,
    epoch (if given), worst-violator instance index, actual value, bound,
    and total violator count.
    """
    cos = stats.cos_per_inst.detach()
    cos_violators = cos < (stats.cos_lower_bound - cos_slack)
    n_cos_violators = int(cos_violators.sum().item())
    if n_cos_violators > 0:
        worst_idx = int(cos.argmin().item())
        ctx = f"[split={label}"
        if epoch is not None:
            ctx += f", epoch={epoch}"
        ctx += f", instance={worst_idx}, n_violators={n_cos_violators}]"
        raise RuntimeError(
            f"Theorem 1 cosine bound violated {ctx}: "
            f"cos<c_tilde, c_hat> = {float(cos[worst_idx].item()):.6f}, "
            f"required >= {stats.cos_lower_bound:.6f} "
            f"(= sqrt(1 - eps^2) - {cos_slack:g})"
        )

    rmse_violation = (stats.rmse_delta_per_inst - stats.rmse_bound_per_inst).detach()
    rmse_violators = rmse_violation > rmse_slack
    n_rmse_violators = int(rmse_violators.sum().item())
    if n_rmse_violators > 0:
        worst_idx = int(rmse_violation.argmax().item())
        actual_delta = float(stats.rmse_delta_per_inst[worst_idx].item())
        bound = float(stats.rmse_bound_per_inst[worst_idx].item())
        ctx = f"[split={label}"
        if epoch is not None:
            ctx += f", epoch={epoch}"
        ctx += f", instance={worst_idx}, n_violators={n_rmse_violators}]"
        raise RuntimeError(
            f"Theorem 1 RMSE bound violated {ctx}: "
            f"RMSE(c_tilde, c) - RMSE(c_hat, c) = {actual_delta:.6e}, "
            f"required <= {bound:.6e} "
            f"(= (eps/sqrt(d)) * ||c_hat||_2 + {rmse_slack:g})"
        )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    F_theta: PerAssetBiasCorrectionLayer,
    X_train: torch.Tensor,         # (N, n_assets, n_features)  scaled
    c_hat_train: torch.Tensor,     # (N, n_assets)
    c_true_train: torch.Tensor,    # (N, n_assets)
    Sigma_static: torch.Tensor,    # (n_assets, n_assets)
    optimizer: torch.optim.Optimizer,
    loss_fn: MarkowitzRegretLoss,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    F_theta.train()
    N = X_train.shape[0]
    perm_np = rng.permutation(N)
    perm = torch.from_numpy(perm_np)
    total_regret = 0.0
    n_batches = 0
    for start in range(0, N, batch_size):
        idxs = perm[start: start + batch_size]
        X_b = X_train[idxs]
        c_hat_b = c_hat_train[idxs]
        c_true_b = c_true_train[idxs]
        cache_idx = idxs.to(torch.int64)

        c_tilde_b = F_theta(X_b, c_hat_b)
        loss = loss_fn(c_tilde_b, c_true_b, Sigma_static, cache_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_regret += float(loss.item())
        n_batches += 1
    return total_regret / max(n_batches, 1)


def evaluate_split(
    F_theta: PerAssetBiasCorrectionLayer,
    X: torch.Tensor,                     # (N, n_assets, n_features)
    c_hat: torch.Tensor,                 # (N, n_assets)
    c_true: torch.Tensor,                # (N, n_assets)
    Sigma_static: torch.Tensor,          # (n_assets, n_assets)
    static_solver: MarkowitzStatic,
    gamma: float,
    epsilon: float,
    n_assets: int,
) -> dict:
    """Forward F_θ on a split, compute regret + Theorem 1 stats + MSEs.

    Both ``ndr_dff`` and ``ndr_two_stage`` are computed against the **same
    Sigma_static** — see docs/exp02_design.md D2 (apples-to-apples by
    construction; the exp01b two-stage NDR used per-instance Σ and is
    therefore not directly comparable).

    Uses ``MarkowitzStatic`` for ``w*(c_tilde)`` / ``w*(c_hat)`` /
    ``w*(c_true)`` (no gradients needed in eval; faster than DiffMarkowitz).
    """
    F_theta.eval()
    with torch.no_grad():
        c_tilde = F_theta(X, c_hat)

    c_true_np = c_true.cpu().numpy()
    c_hat_np = c_hat.cpu().numpy()
    c_tilde_np = c_tilde.cpu().numpy()
    Sigma_np = Sigma_static.cpu().numpy()

    # Both NDRs share Sigma_static — see docs/exp02_design.md D2.
    w_oracle = static_solver.solve_batch(c_true_np, Sigma_np)
    w_dff = static_solver.solve_batch(c_tilde_np, Sigma_np)
    w_two_stage = static_solver.solve_batch(c_hat_np, Sigma_np)

    def f(c, w):
        linear = -(c * w).sum(axis=-1)
        quad = np.einsum("bi,ij,bj->b", w, Sigma_np, w)
        return linear + gamma * quad

    f_oracle = f(c_true_np, w_oracle)
    f_dff = f(c_true_np, w_dff)
    f_two_stage = f(c_true_np, w_two_stage)

    regret_dff = float((f_dff - f_oracle).mean())
    regret_two_stage = float((f_two_stage - f_oracle).mean())
    ndr_dff = float((f_dff - f_oracle).sum() / max(np.abs(f_oracle).sum(), 1e-8))
    ndr_two_stage = float(
        (f_two_stage - f_oracle).sum() / max(np.abs(f_oracle).sum(), 1e-8)
    )

    mse_chat = float(((c_hat_np - c_true_np) ** 2).mean())
    mse_ctilde = float(((c_tilde_np - c_true_np) ** 2).mean())

    stats = theorem1_diagnostics(c_tilde, c_hat, c_true, epsilon, n_assets)
    return {
        "regret_dff": regret_dff,
        "regret_two_stage": regret_two_stage,
        "ndr_dff": ndr_dff,
        "ndr_two_stage": ndr_two_stage,
        "mse_chat": mse_chat,
        "mse_ctilde": mse_ctilde,
        "cos_mean": float(stats.cos_per_inst.mean().item()),
        "cos_min": float(stats.cos_per_inst.min().item()),
        "cos_lower_bound": float(stats.cos_lower_bound),
        "rmse_delta_mean": float(stats.rmse_delta_per_inst.mean().item()),
        "rmse_delta_max": float(stats.rmse_delta_per_inst.max().item()),
        "rmse_bound_mean": float(stats.rmse_bound_per_inst.mean().item()),
        "_t1_stats": stats,  # for assertion outside
    }


# ---------------------------------------------------------------------------
# Argparse + run dir + logging setup (CLI plumbing)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon", type=float, default=None,
                   help="Override config epsilon (Phase 3 sweep entry point).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override the autogenerated runs/exp02_dff_<ts>/ path.")
    return p.parse_args()


def make_run_dir(override: Optional[str]) -> Path:
    if override is not None:
        run_dir = Path(override)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"runs/exp02_dff_{ts}")
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Optional[Path]) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger("exp02")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "log.txt", mode="w")
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# run_trainer — pure callable, no argparse / loader I/O. Used by main() and tests.
# ---------------------------------------------------------------------------
def run_trainer(
    config: dict,
    instances: list[Instance],
    run_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Run a single DFF training pass on the given instances + config.

    Parameters
    ----------
    config
        Output of :func:`base_config` (possibly with overrides). Must
        contain at least: epsilon, gamma, lr, epochs, batch_size, seed,
        hidden_dim, n_hidden_layers, n_cross_fit_splits, val_fraction,
        embargo_days, test_year, xgb_kwargs.
    instances
        List of :class:`Instance` already loaded (the caller handles
        :class:`PortfolioDataLoader` invocation; this function does not).
    run_dir
        Directory to save artifacts to. ``None`` skips disk I/O —
        useful for tests.
    logger
        Optional pre-configured logger. Defaults to a stdout-only one.

    Returns
    -------
    dict with keys:
        ``epoch_metrics``: list of per-epoch dicts
        ``final``:        the final test metrics dict (also written to
                          ``test_metrics.json`` if ``run_dir`` is given)
        ``n_train_inner``, ``n_val``, ``n_test``: split sizes
    """
    if logger is None:
        logger = logging.getLogger("exp02_run_trainer")
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        logger.setLevel(logging.INFO)

    timing: dict[str, float] = {}
    _t_total_start = time.perf_counter()

    # Determinism: set every relevant RNG seed up-front, then construct one
    # numpy Generator for batch shuffling (passed into train_one_epoch).
    set_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])

    logger.info("config (excl. xgb_kwargs):\n%s", json.dumps(
        {k: v for k, v in config.items() if k != "xgb_kwargs"},
        indent=2, default=str,
    ))

    # --- 1. Split instances ---
    train_all, test = PortfolioDataLoader.split(
        instances, test_year=config["test_year"], embargo_days=config["embargo_days"]
    )
    train_inner, val_set = time_ordered_val_split(train_all, config["val_fraction"])
    n_assets = train_inner[0].X.shape[0]
    n_features = train_inner[0].X.shape[1]
    logger.info(
        "split: train_inner=%d, val=%d, test=%d (n_assets=%d, n_features=%d)",
        len(train_inner), len(val_set), len(test), n_assets, n_features,
    )
    logger.info(
        "date ranges: train_inner=[%s, %s], val=[%s, %s], test=[%s, %s]",
        train_inner[0].metadata["date"].date(), train_inner[-1].metadata["date"].date(),
        val_set[0].metadata["date"].date(), val_set[-1].metadata["date"].date(),
        test[0].metadata["date"].date(), test[-1].metadata["date"].date(),
    )

    # --- 2. Cross-fit on train_inner for OOF c_hat ---
    logger.info("cross-fitting backbone on train_inner (n_splits=%d)",
                config["n_cross_fit_splits"])
    _t = time.perf_counter()
    c_hat_train_np = cross_fit_predict(
        XGBRegressor, config["xgb_kwargs"], train_inner,
        n_splits=config["n_cross_fit_splits"],
    )
    timing["cross_fit_seconds"] = time.perf_counter() - _t
    logger.info("cross-fit done in %.1fs; c_hat_train shape=%s",
                timing["cross_fit_seconds"], c_hat_train_np.shape)

    # --- 3. Refit M_full on train_inner for val/test inference (D4) ---
    logger.info("refitting M_full on full train_inner")
    _t = time.perf_counter()
    M_full = XGBRegressor(**config["xgb_kwargs"])
    M_full.fit(panel_X(train_inner), np.concatenate([i.c_true for i in train_inner]))
    timing["m_full_fit_seconds"] = time.perf_counter() - _t
    c_hat_val_np = predict_panel(M_full, val_set)
    c_hat_test_np = predict_panel(M_full, test)
    logger.info("M_full fit done in %.1fs; predictions: val=%s, test=%s",
                timing["m_full_fit_seconds"], c_hat_val_np.shape, c_hat_test_np.shape)

    # --- 4. Static Sigma (D2a) — LedoitWolf on the train_inner forward-return panel ---
    c_true_train_np = panel_c_true(train_inner)
    Sigma_static_np = LedoitWolf().fit(c_true_train_np).covariance_
    Sigma_static_np = 0.5 * (Sigma_static_np + Sigma_static_np.T)  # symmetrize defensively
    logger.info("Sigma_static shape=%s, trace=%.6e",
                Sigma_static_np.shape, float(np.trace(Sigma_static_np)))

    # --- 5. Oracle cache via MarkowitzStatic ---
    static_solver = MarkowitzStatic(
        n_assets=n_assets, risk_aversion=config["gamma"], long_only=True,
    )
    logger.info("building oracle cache (MarkowitzStatic.solve_batch)")
    _t = time.perf_counter()
    w_oracle_train = build_oracle_cache(
        c_true_train_np, Sigma_static_np, gamma=config["gamma"],
        solver_static=static_solver,
    )
    logger.info("oracle cache done in %.1fs", time.perf_counter() - _t)

    # --- 6. Standardize features ---
    X_train_panel = panel_X(train_inner)
    scaler = StandardScaler().fit(X_train_panel)
    X_train_scaled = reshape_panel_to_per_asset(
        scaler.transform(X_train_panel), n_assets
    )
    X_val_scaled = reshape_panel_to_per_asset(
        scaler.transform(panel_X(val_set)), n_assets
    )
    X_test_scaled = reshape_panel_to_per_asset(
        scaler.transform(panel_X(test)), n_assets
    )

    def t64(arr) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float64)

    X_train_t = t64(X_train_scaled)
    X_val_t = t64(X_val_scaled)
    X_test_t = t64(X_test_scaled)
    c_hat_train_t = t64(c_hat_train_np)
    c_hat_val_t = t64(c_hat_val_np)
    c_hat_test_t = t64(c_hat_test_np)
    c_true_train_t = t64(c_true_train_np)
    c_true_val_t = t64(panel_c_true(val_set))
    c_true_test_t = t64(panel_c_true(test))
    Sigma_static_t = t64(Sigma_static_np)

    # --- 7. Build F_θ (D1b) ---
    F_theta = PerAssetBiasCorrectionLayer(
        n_features_per_asset=n_features,
        epsilon=config["epsilon"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_hidden_layers"],
    ).double()
    logger.info(
        "F_theta: PerAssetBiasCorrectionLayer(F=%d, eps=%.3f, hidden=%d, layers=%d) — "
        "%d trainable params",
        n_features, config["epsilon"], config["hidden_dim"],
        config["n_hidden_layers"],
        sum(p.numel() for p in F_theta.parameters() if p.requires_grad),
    )

    # --- 8. Loss + optimizer ---
    diff_solver = DiffMarkowitz(gamma=config["gamma"])
    loss_fn = MarkowitzRegretLoss(
        gamma=config["gamma"], diff_markowitz=diff_solver, w_oracle_cache=w_oracle_train,
    )
    optimizer = torch.optim.Adam(F_theta.parameters(), lr=config["lr"])

    # --- 9. Training loop ---
    epoch_metrics: list[dict] = []
    logger.info("starting training: %d epochs, batch_size=%d",
                config["epochs"], config["batch_size"])
    _t_train_loop = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        train_regret = train_one_epoch(
            F_theta, X_train_t, c_hat_train_t, c_true_train_t, Sigma_static_t,
            optimizer, loss_fn, config["batch_size"], rng,
        )
        val_metrics = evaluate_split(
            F_theta, X_val_t, c_hat_val_t, c_true_val_t, Sigma_static_t,
            static_solver, config["gamma"], config["epsilon"], n_assets,
        )
        # Theorem 1 — raise if violated. Checked on val each epoch; on train
        # the bound holds analytically because the layer enforces it
        # elementwise by construction.
        assert_theorem1(val_metrics["_t1_stats"], label="val", epoch=epoch)

        row = {
            "epoch": epoch,
            "train_regret": train_regret,
            "val_regret": val_metrics["regret_dff"],
            "val_regret_two_stage": val_metrics["regret_two_stage"],
            "cos_mean": val_metrics["cos_mean"],
            "cos_min": val_metrics["cos_min"],
            "rmse_delta_mean": val_metrics["rmse_delta_mean"],
            "rmse_delta_max": val_metrics["rmse_delta_max"],
        }
        epoch_metrics.append(row)
        logger.info(
            "epoch %02d/%d | train_regret=%.6e val_regret=%.6e "
            "(two-stage=%.6e) | cos[min=%.4f mean=%.4f bound=%.4f] "
            "| rmse_delta[mean=%.3e max=%.3e]",
            epoch, config["epochs"], train_regret, val_metrics["regret_dff"],
            val_metrics["regret_two_stage"], val_metrics["cos_min"],
            val_metrics["cos_mean"], val_metrics["cos_lower_bound"],
            val_metrics["rmse_delta_mean"], val_metrics["rmse_delta_max"],
        )

    timing["train_loop_seconds"] = time.perf_counter() - _t_train_loop

    # --- 10. Final test eval ---
    logger.info("final test evaluation")
    _t = time.perf_counter()
    test_metrics = evaluate_split(
        F_theta, X_test_t, c_hat_test_t, c_true_test_t, Sigma_static_t,
        static_solver, config["gamma"], config["epsilon"], n_assets,
    )
    timing["final_eval_seconds"] = time.perf_counter() - _t
    assert_theorem1(test_metrics["_t1_stats"], label="test", epoch=None)

    timing["total_seconds"] = time.perf_counter() - _t_total_start

    final = {
        "test_ndr_dff": test_metrics["ndr_dff"],
        "test_ndr_two_stage": test_metrics["ndr_two_stage"],
        "improvement_pp": test_metrics["ndr_two_stage"] - test_metrics["ndr_dff"],
        "test_mse_ctilde": test_metrics["mse_ctilde"],
        "test_mse_chat": test_metrics["mse_chat"],
        "test_cos_mean": test_metrics["cos_mean"],
        "test_cos_min": test_metrics["cos_min"],
        "test_cos_lower_bound": test_metrics["cos_lower_bound"],
        "test_rmse_delta_mean": test_metrics["rmse_delta_mean"],
        "test_rmse_delta_max": test_metrics["rmse_delta_max"],
        "n_train_inner": len(train_inner),
        "n_val": len(val_set),
        "n_test": len(test),
        "timing": timing,
    }
    logger.info("test results:\n%s", json.dumps(final, indent=2))

    # --- 11. Save artifacts (D7) — only if a run_dir was provided ---
    if run_dir is not None:
        (run_dir / "config.json").write_text(json.dumps(
            config, indent=2, default=str
        ))
        with (run_dir / "metrics_per_epoch.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_metrics[0].keys()))
            writer.writeheader()
            for r in epoch_metrics:
                writer.writerow(r)
        (run_dir / "test_metrics.json").write_text(json.dumps(final, indent=2))
        torch.save(F_theta.state_dict(), run_dir / "model.pt")
        joblib.dump(scaler, run_dir / "scaler.pkl")
        M_full.save_model(str(run_dir / "M_full.json"))
        logger.info("artifacts saved under %s", run_dir)

    return {
        "epoch_metrics": epoch_metrics,
        "final": final,
        "n_train_inner": len(train_inner),
        "n_val": len(val_set),
        "n_test": len(test),
    }


# ---------------------------------------------------------------------------
# main() — CLI wrapper, loads instances, then delegates to run_trainer
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config = base_config()
    if args.epsilon is not None:
        config["epsilon"] = float(args.epsilon)
    if args.seed is not None:
        config["seed"] = int(args.seed)
        config["xgb_kwargs"]["random_state"] = config["seed"]   # keep in sync
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir

    run_dir = make_run_dir(args.output_dir)
    config["output_dir"] = str(run_dir)
    log = setup_logging(run_dir)
    log.info("exp02 DFF run starting | run_dir=%s", run_dir)

    loader = PortfolioDataLoader(
        tickers=config["universe_tickers"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        cov_window=config["cov_window"],
        cache_dir=config["cache_dir"],
    )
    instances = loader.load()
    log.info("loaded %d instances from PortfolioDataLoader", len(instances))

    run_trainer(config, instances, run_dir=run_dir, logger=log)


if __name__ == "__main__":
    main()
