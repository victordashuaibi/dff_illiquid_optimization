# exp02 — DFF Trainer Design Decisions

This doc locks in the seven open design decisions (D1–D7) for
`experiments/exp02_dff_train.py` *before* writing any trainer code.
Every divergence from the DFF paper (Yang et al. AAAI 2025) is
flagged inline so future-me can find it.

## D1 — F_θ input encoding

**Decision: D1b (per-asset shared weights), with the per-asset NN input
augmented by the per-asset scalar ĉ_i.**

For an instance with feature matrix `X ∈ ℝ^{N × F}` and backbone
prediction `ĉ ∈ ℝ^N`, F_θ is applied row-by-row with shared weights:
each asset row sees `(features_i, ĉ_i) ∈ ℝ^{F+1}` as input and
produces a scalar `φ_i`. The corrected prediction is
`c̃_i = φ_i · ĉ_i`, with `φ_i ∈ [1−ε, 1+ε]` enforced by the same
sigmoid bound as the existing `BiasCorrectionLayer`.

Considered:

| Option | Pros | Cons |
|---|---|---|
| D1a — flatten `(N·F,)`, output `N` | Simplest; one NN call per instance | Loses asset-permutation equivariance — the network "remembers" the alphabetical asset ordering, which has no semantic content on Russell ILLIQ-30 |
| **D1b — per-asset shared NN, input `(F+1,)`** | Permutation-equivariant; same NN across assets; scales to any `N` | Slight deviation from paper Eq. 11 which writes `h(x)`, not `h(x, ĉ)` |
| D1c — flatten `(N·F + N,)` (concat ĉ to flattened X) | Lets the NN see ĉ explicitly | Same equivariance loss as D1a, plus extra dim |

**Why D1b**: ILLIQ-30 microcaps are asset-fungible — the model has no
reason to treat ticker LWAY differently from USAU based on ordering.
D1a/D1c break that symmetry; D1b keeps it. The paper's `h(x)` is
agnostic — passing per-asset `(features_i, ĉ_i)` is consistent with the
spirit of Eq. 11 even though the literal x is enriched.

**Paper deviation**: Paper has `h(x)`, our D1b uses `h(x, ĉ)`. This is
the user-recommended form per the Task-4 prompt. Document in the paper
write-up under "implementation choices" if/when we publish.

**Implementation**: new `PerAssetBiasCorrectionLayer` in
`src/dff/bias_correction.py` that wraps the existing
`BiasCorrectionLayer` (output_dim=1) and applies it row-wise. Existing
layer is not modified; the wrapper just reshapes before/after.

## D2 — Sigma strategy

**Decision: D2a — static Ledoit-Wolf shrinkage covariance estimated on
the train_inner forward-return panel, used identically at oracle-cache
build time and at training time.**

Considered:

| Option | Pros | Cons |
|---|---|---|
| **D2a — static Σ across the run** | One DiffMarkowitz call per batch (no rebuild); fast; consistent oracle vs. train | No time-variation in risk |
| D2b — per-instance rolling Σ (already produced by the loader), serialized through DiffMarkowitz one row at a time | Honest; matches `MarkowitzStatic` baseline | DiffMarkowitz rebuilds its CvxpyLayer on every Σ change → 80k cvxpylayers solves per training run at B=32, 50 epochs. Hours-to-days runtime. |
| D2c — per-instance Σ batched via a Σ-as-cvxpy-Parameter `DiffMarkowitz` | Honest and fast | Requires modifying `DiffMarkowitz`; out-of-scope for Task 4 |

**Why D2a**: We need a working DFF run end-to-end first. D2c is the
right long-term answer but requires a non-trivial refactor of
`DiffMarkowitz` (Σ becomes a `cp.Parameter` so DPP holds, batched
solves work natively); blocking on that delays everything. D2b is too
slow to iterate on. D2a gets us a runnable trainer today. We document
that the static-Σ run is a *methodological simplification of the
exp01b baseline*, which uses per-instance Σ — so the two are not
directly comparable on identical optimization problems.

**Σ source**: `LedoitWolf().fit(c_true_train_inner).covariance_`. This
is the covariance of *21-day forward returns* across train_inner
instances — exactly the quantity that `w'Σw` represents in the
penalty-form Markowitz objective. Not daily-return covariance, not the
loader's per-instance trailing-60d Σ.

**Cache–training consistency**: the same `Σ_static` numpy array is
passed to `build_oracle_cache` (via `MarkowitzStatic`) and to
`MarkowitzRegretLoss` at every train/val/test step. If they ever
diverge, regret is no longer non-negative.

**Future**: Task 5 sweep can also tune ε under D2a. Task 7+ should
move to D2c by lifting Σ into a cvxpy Parameter inside `DiffMarkowitz`,
then re-running.

## D3 — Train/val split

**Decision: last 10% of `train` (after the embargo cut), time-ordered,
becomes `val`. The remaining 90% is `train_inner`. Cross-fitting
operates on `train_inner` only; `M_full` is also fit on `train_inner`
only. Val is then predicted by `M_full`.**

```
[ year < test_year, embargo cut ]   →   train (90% inner) + val (last 10%)
                                                    ▲ time
[ year == test_year, embargo cut ]  →   test
```

Considered:

| Option | Pros | Cons |
|---|---|---|
| **Last 10% of train, time-ordered** | Simple; matches "what does F_θ generalize to next?" | 10% of ~1100 train instances ≈ 110 days ≈ 5 months — usable, not huge |
| Inner 2-fold (use one cross-fit fold as val) | Couples val sample size to cross-fit; statistically cleaner | Couples val to fold structure; messy when n_splits != 2; no clean way to time-order val |

**Why last 10%**: simplicity wins here. Val sits *after* both cross-fit
folds in calendar time (because it's the most recent slice of train),
so its predictions are honest OOF predictions from `M_full` trained on
strictly earlier data.

**No additional embargo from val→test**: `split()` already cuts an
embargo between train (which val is part of) and test. The last day of
val is at most `embargo_days` trading days before the first day of
test — which is the embargo we already chose. No change needed.

**No embargo from train_inner→val**: not needed for the same
information-leakage reason: there's no forward-looking target leakage
in the `train_inner → val` direction *for the F_θ training*, because
F_θ trains on `c_hat_train_inner` (cross-fit predictions, leak-free by
construction) and is evaluated on `c_hat_val` (predictions from
`M_full` trained on `train_inner` only — `M_full` never saw val data,
so val labels and val features are both unseen).

The only residual leak path is rolling features at the
train_inner→val boundary using ≤20-day-old prices. Worth flagging but
small; the embargo on the test side already pays for the analogous
leak. We accept this for now.

## D4 — When to refit M for test prediction

**Decision: refit one `M_full` (XGBRegressor with the same kwargs as
the cross-fit folds) on the panel-flattened `train_inner` after
cross-fitting completes. Use `M_full` for both val and test
predictions.**

Three M-shaped objects exist over a run:

1. `M_fold1`, `M_fold2` — created and discarded inside
   `cross_fit_predict` to produce `c_hat_train_inner` (OOF, leak-free).
2. `M_full` — trained once on all of `train_inner`, persisted to disk,
   used for `c_hat_val = M_full.predict(val)` and
   `c_hat_test = M_full.predict(test)`.

The cross-fit Ms are not used for any inference outside of cross-fitting
itself; that would be a leakage path.

`M_full.save_model(...)` stores the booster as JSON for reproducibility.

## D5 — Hyperparameters

Locked in a single dict at the top of the trainer. Anything that
diverges from paper §6.1 has an inline comment.

```python
CONFIG = {
    # ---- data ----
    "tickers_source": "russell_illiq_30",   # exp01b universe
    "start_date":   "2018-01-01",
    "end_date":     "2023-12-31",
    "test_year":    2023,
    "n_keep":       30,

    # ---- F_θ (paper §6.1) ----
    "hidden_dim":      32,
    "n_hidden_layers": 3,
    "epsilon":         0.5,                 # paper synthetic default; will sweep in Task 6

    # ---- optimizer (paper §6.1) ----
    "lr":         1e-3,
    "batch_size": 32,
    "epochs":     50,

    # ---- our additions ----
    "gamma":            1.0,                # match exp01b risk_aversion
    "embargo_days":     None,               # auto-derive from features module
    "n_cross_fit_splits": 2,                # paper default
    "val_fraction":     0.10,
    "seed":             42,

    # ---- backbone (legacy from exp01b) ----
    "xgb_kwargs": {                         # match exp01b for direct comparability
        "objective":         "reg:squarederror",
        "n_estimators":      800,
        "learning_rate":     0.03,
        "max_depth":         4,
        "min_child_weight":  5,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        2.0,
        "random_state":      42,
        "n_jobs":            -1,
        "tree_method":       "hist",
    },

    # ---- runtime ----
    "cache_dir":  "data/processed",
    "output_dir": None,                     # filled in at runtime: runs/exp02_dff_<ts>/
}
```

Anything not listed (e.g. `gamma_scheduler`, `weight_decay`,
`grad_clip`) is intentionally not added — paper doesn't use them; if
training is unstable we add them in Task 5+ with explicit justification.

## D6 — What to log

### Per epoch

| Field | How |
|---|---|
| `epoch` | int, 1-indexed |
| `train_regret` | mean over train batches (the loss objective F_θ is being trained on) |
| `val_regret` | F_θ → DiffMarkowitz/MarkowitzStatic → regret on val, with `torch.no_grad()` |
| `cos_mean` | mean over val of `cos⟨c̃_i, ĉ_i⟩` |
| `cos_min` | min over val instances of `cos⟨c̃_i, ĉ_i⟩` |
| `rmse_delta_mean` | mean over val of per-instance `RMSE(c̃, c) − RMSE(ĉ, c)` |
| `rmse_delta_max` | max over val of the same |

### Theorem 1 assertions (per epoch, after val pass)

Both raise `RuntimeError` with full context if violated:

```
cos_min            >=  sqrt(1 - epsilon^2) - 1e-3        # Eq. 14
rmse_delta_max     <=  (epsilon/sqrt(d)) * mean_norm_chat + 1e-3   # Eq. 12
```

I check `cos_min` and `rmse_delta_max` (not means) because the bound
is per-instance — a single violator is a real bug. The 1e-3 slack
absorbs cvxpylayers solver noise. Mean fields are still logged for
trend visibility.

### Per run (final, on test)

| Field | Notes |
|---|---|
| `test_ndr_dff` | Markowitz NDR using F_θ output → MarkowitzStatic on c̃ |
| `test_ndr_two_stage` | NDR using bare M_full output → MarkowitzStatic on ĉ. **This is the new headline baseline** (NDR=80.08% from the embargo'd exp01b run; two-stage with static Σ may differ slightly because exp01b uses per-instance Σ — flag in the report) |
| `improvement_pp` | `test_ndr_two_stage − test_ndr_dff`, signed |
| `test_mse_ctilde` | MSE(c̃, c) on test |
| `test_mse_chat` | MSE(ĉ, c) on test (sanity check that M_full's MSE matches the embargo'd two-stage result) |
| `test_cos_mean` | Theorem-1 cos⟨c̃, ĉ⟩ on test |
| `test_rmse_delta_mean` | Theorem-1 RMSE delta on test |

## D7 — Run artifact layout

```
runs/exp02_dff_<YYYYMMDD_HHMMSS>/
    config.json              # full resolved CONFIG dict (with derived embargo, etc.)
    metrics_per_epoch.csv    # epoch,train_regret,val_regret,cos_mean,cos_min,rmse_delta_mean,rmse_delta_max
    test_metrics.json        # final per-run numbers (D6 last table)
    model.pt                 # F_θ state_dict (PerAssetBiasCorrectionLayer)
    scaler.pkl               # joblib.dump of the StandardScaler fit on train_inner X
    M_full.json              # XGBRegressor.save_model(...) JSON
    log.txt                  # full stdout/stderr captured for the run
```

**Naming**: timestamp is UTC, `YYYYMMDD_HHMMSS`. Task 6 (sweep) will
write siblings under the same `runs/` parent — for the sweep they'll
nest under `runs/exp02_eps_sweep_<ts>/eps_<value>/...`. Out of scope
for Task 4.

**What's NOT saved**: per-batch loss curves (use the per-epoch CSV
plus the log file), the oracle cache (deterministic from
`(c_true_train_inner, Σ_static, γ)` so re-derivable), and the
`DiffMarkowitz` instance (parameter-free; identity is just `gamma` +
the cached `Σ_static`).

## Open question deferred to Task 5

None right now. If something turns out ambiguous when I write the
trainer, I will stop and ask before guessing.
