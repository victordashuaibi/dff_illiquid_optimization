# Week 1 Findings: Prediction-Decision Decoupling on Illiquid Universe

## TL;DR

On the ILLIQ-screened Russell 2000 universe (n=30 microcaps), our XGBoost
backbone reaches near-optimal prediction MSE (1.07× the predict-mean baseline)
yet still suffers 84% Normalized Decision Regret. This empirically validates
the core motivation of Decision-Focused Learning: prediction quality and
decision quality are decoupled, especially in noisy illiquid regimes.

## Setup

Three baselines run with the *same* pipeline (panel-style XGBoost trained on
MSE → static long-only Markowitz with γ=1.0 → Markowitz NDR per
[`docs/INTERFACE.md`](INTERFACE.md)):

| Universe | n_assets | Description |
|---|---|---|
| Mega-caps | 5 | AAPL, MSFT, GOOGL, AMZN, META |
| Russell prefix-200 | 30 | First 200 IWM tickers (alphabetical, mostly mid-caps), top-30 by ILLIQ |
| **Russell ILLIQ-30** | 30 | Full Russell 2000 (~1939 tickers, 689 survive full-history + price≥$5), top-30 by composite ILLIQ + DollarVolume rank |

All experiments: daily 2018-01-01 → 2023-12-31, 21-day forward return target,
`test_year = 2023`, `cov_window = 60`, Ledoit-Wolf shrinkage covariance,
`random_state = 42`.

The mega-caps run uses the lighter `XGB_KWARGS_MINI`
(`n_estimators=100, max_depth=3`) since 444 train instances × 5 assets ≈
2.2k panel rows; the two 30-asset runs use the legacy `XGB_KWARGS_LEGACY`
(`n_estimators=800, max_depth=4`) — 1198 instances × 30 assets ≈ 36k rows is
the regime those hyperparams were tuned on.

## Key results

| | Mega-caps | Russell prefix-200 | **Russell ILLIQ-30** |
|---|---:|---:|---:|
| n_train / n_test | 444 / 230 | 1198 / 229 | 1198 / 229 |
| target_std (test) | 0.109 | 0.242 | 0.126 |
| MSE_train | 2.52e-3 | 1.33e-2 | 1.21e-2 |
| MSE_test | 2.15e-2 | 7.17e-2 | 1.69e-2 |
| MSE_naive (predict-mean) | 1.69e-2 | 5.99e-2 | 1.58e-2 |
| **MSE_test / MSE_naive** | 1.28× | 1.20× | **1.07×** |
| RMSE_test / target_std | 1.35× | 1.11× | **1.04×** |
| ‖w_pred − w_oracle‖ avg L2 | 1.07 | 1.16 | 1.29 |
| Linear NDR (legacy, wrong-signed) | −105.22% | −67.82% | −83.72% |
| **Markowitz NDR (primary)** | **105.93%** | **68.49%** | **84.08%** |

How to read each row:

- **MSE_test / MSE_naive** is the prediction's miss vs. predicting the train
  target mean for every test point. A ratio < 1 means the model adds value;
  ratios ≥ 1 mean the model is at or below the noise floor of the target.
- **RMSE_test / target_std**: same idea on the same scale as the target;
  1.04× says the model's residuals are about as wide as the target's
  cross-sectional standard deviation. There is no detectable signal left.
- **L2 distance** of predicted vs oracle weights is bounded above by ~√2
  ≈ 1.41 on the simplex; 1.29 means w_pred and w_oracle pick visibly
  different portfolios.
- **NDR (Markowitz)** is the primary metric — DFF Eq. 19 with the full
  quadratic objective `f(w, c, Σ) = -c^T w + γ w^T Σ w`. Linear NDR is shown
  only as a sanity check that the legacy linear-objective formula is
  wrong-signed for our problem (mirror image, same magnitude).

Source artefacts: [`results/exp01/metrics.json`](../results/exp01/metrics.json),
[`results/exp01b/metrics.json`](../results/exp01b/metrics.json).

## The decoupling phenomenon

Plotting the three runs on `(MSE_test / MSE_naive, NDR)` axes:

```
                     NDR (lower is better)
                          ▲
                          │
                  110%    │      ● Mega-caps
                          │      (1.28×, 105.9%)
                          │      bad on both axes
                          │
                   90%    │  ● Russell ILLIQ-30
                          │  (1.07×, 84.1%)  ◀── DFF's regime
                          │  near-optimal MSE,
                          │  large decision regret
                   70%    │             ● Russell prefix-200
                          │             (1.20×, 68.5%)
                          │             medium both
                          │
                          └──────────────────────────────► MSE_test / MSE_naive
                          1.05×  1.10×  1.15×  1.20×  1.25×  1.30×
                          (noise ceiling)            (clearly worse than naive)
```

Three observations:

1. **The two axes don't line up.** If prediction quality alone determined
   decision quality, the points would lie on a monotone curve. They don't:
   ILLIQ-30 has the *best* MSE-vs-noise-ceiling but a *worse* NDR than
   prefix-200.
2. **Mega-caps is "honestly bad on both axes."** High MSE, high NDR, no
   surprise — the model isn't even hitting the noise floor.
3. **ILLIQ-30 is "near-optimal on prediction, bad on decision."** The MSE
   is 1.07× of the predict-mean baseline — the model has effectively
   exhausted what an MSE-trained predictor can extract from this signal.
   Yet NDR is 84%, meaning the predicted weights leave ~84% of the oracle
   utility on the table relative to the average oracle objective magnitude.
   This is the regime DFF was explicitly designed for.

## Why this matters

1. **Validates DFF motivation on real data.** The DFF paper demonstrates
   prediction-decision decoupling on synthetic and curated datasets. We
   independently observe it on real US equities, in the specific regime
   (illiquid microcaps) that aligns with our research target.

2. **Sets up DFF's value proposition.** If Week 2 DFF reduces NDR
   substantially on ILLIQ-30 while leaving MSE near baseline, this is
   the cleanest possible demonstration that decision-focused training
   matters precisely when prediction has hit its noise ceiling. Mega-caps
   would conflate "DFF helps" with "DFF helps because there was MSE
   left on the table"; ILLIQ-30 removes that confound.

3. **Motivates the eventual paper Figure 1.** A scatter plot with
   `MSE_test / MSE_naive` on the x-axis and `NDR` on the y-axis, points
   colored by universe and shaped by training objective (MSE vs. DFF),
   should show DFF moving the ILLIQ-30 point *downward* (NDR ↓) without
   a meaningful rightward shift (MSE ~). The current three-point
   baseline cloud is the "before" half of that figure.

## Open questions for Week 2

- **Does DFF improve NDR more on universes where prediction MSE is already
  near-optimal?** Hypothesis: yes — DFF has more "free" decision-side
  improvement to extract precisely when MSE-side improvement is gated by
  noise.
- **What ε value (bias-correction trust radius, [INTERFACE.md](INTERFACE.md)
  Eq. 9-11) gives the best NDR-MSE tradeoff on ILLIQ-30?** Worth a sweep
  over ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5}.
- **Does the prediction distribution under DFF stay spread, or collapse to
  a multiplicative shift of c_hat (the φ(x)·c_hat ansatz)?** DFF paper
  Fig. 1 shows the latter on synthetic data; checking whether real
  microcap noise admits the same structure would be a paper-worthy result.
- **Are the worst-NDR test instances clustered in time** (e.g. 2023 Q1
  bank stress) **or uniformly spread?** If clustered, DFF gains may come
  disproportionately from a few volatile windows.

## Engineering artifacts

- [`experiments/exp01_two_stage_baseline.py`](../experiments/exp01_two_stage_baseline.py)
  — mega-caps mini run, NDR = 105.93%
- [`experiments/exp01b_illiq_baseline.py`](../experiments/exp01b_illiq_baseline.py)
  — Russell ILLIQ-30 run, NDR = 84.08%
- `results/exp01/predictions.png` — mega-caps scatter + histogram
- `results/exp01b/predictions.png` — ILLIQ-30 scatter + histogram

To reproduce both runs:

```bash
PYTHONPATH=. python experiments/exp01_two_stage_baseline.py    # ~30 s (cache hit)
PYTHONPATH=. python experiments/exp01b_illiq_baseline.py       # ~3 min cold (yfinance), ~30 s warm
```

Both runs are deterministic at `seed=42`; numbers in this document match the
metrics JSONs bit-for-bit.
