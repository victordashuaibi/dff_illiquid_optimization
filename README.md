# DFF for Illiquid Asset Portfolio Optimization

Decision-Focused Fine-Tuning (DFF) for portfolio optimization with
illiquid assets, multi-objective considerations (return, liquidity, ESG),
and limited data regimes.

Based on:
- Wilder et al. (AAAI 2019) — Decision-Focused Learning
- Yang et al. (AAAI 2025) — Decision-Focused Fine-Tuning

## Project status
Feasibility / replication phase.

## Setup

```bash
git clone https://github.com/victordashuaibi/dff_illiquid_optimization.git
cd dff_illiquid_optimization
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Repo structure

```
src/
  backbone/    # XGBoost, NN predictive models (MSE-trained)
  optimizer/   # cvxpylayers-based portfolio QP
  dff/         # Bias correction layer F_theta
  losses/      # SPO+, decision regret, NDR
  data/        # Data loaders for FINRA bonds, Yahoo equities
  utils/
experiments/   # Runnable experiment scripts
notebooks/     # Exploratory analysis, plots
configs/       # YAML configs
tests/         # Unit tests
```

## Data
Raw data lives in a shared Google Drive folder (not in git).

## Roadmap
- [ ] Week 1-2: Two-stage baselines + toy LP DFL replication
- [ ] Week 3: First end-to-end DFF on Yahoo equities
- [ ] Week 4: Bond data integration + epsilon sensitivity
- [ ] Week 5-6: Feasibility report
