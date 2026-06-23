# Diffusion-driven hedging experiments handoff

This branch adds the diffusion-driven portfolio hedging application that replicates
Cont & Vuletić (2025) on SPX options, using the team's diffusion model as a black-box
scenario generator. The classical layer (Black-Scholes pricing, the LASSO local-risk
solver, AIC alpha selection, and the delta / delta-vega baselines) is reused unchanged;
only the scenario generator is swapped from the VolGAN stand-in to the diffusion model.

## What changed

New source files:

| File | Purpose |
|---|---|
| `backtest_diffusion.py` | Diffusion-LASSO hedging backtest driver; writes the per-observation CSV that the tables and figures derive from |
| `diffusion_hedging_adapter.py` | Wraps the trained diffusion checkpoint behind the same interface as `volgan_adapter.py` |
| `diffusion_state.py` | Per-date conditioning lookup (rolling 21-day history of 11x9 surfaces) from the processed shared-grid tensor |
| `eval_hedging.py` | Builds Table 2, Tables 1/3/4, and Figures 3–14 from the backtest CSV |
| `eval_data_figures.py` | Data-descriptive Figures 1–2 (bid-ask spread surfaces, ATM spread vs arbitrage) |

Modified files:

- `backtest_volgan.py` — paper-faithful delta-vega baseline (fixed K=S0 option held to
  expiry, vega floor, delta off-by-one fix). The VolGAN path is deprecated; kept for
  reference.
- `prepare_shared_grid_data.py` — output filenames parameterized by `seq_len` (supports
  the corrected `seq_len=22` alongside the legacy 30-day data).
- `.gitignore` — resolved a stale merge conflict; ignores the large/regenerable
  artifacts (`model_results/`, `scenarios/`, `*.npy`, `*.pt`, `eval/` outputs) while
  committing `results/`.

## How to run

```bash
conda activate diffusion

# Full backtest over the paper m0 grid (first pass fills the scenario cache, ~1.7h;
# cache reruns are instant). Instrument set is +underlying by default (paper §4.3).
python backtest_diffusion.py \
  --checkpoint model_results/dfm_shared_grid_22d_logiv_call_return_ts1779948122_seed42/model-epoch-3000.pt \
  --all-m0 --n-scenarios 1000 --n-val 100 --sampling-timesteps 50 \
  --scenario-cache scenarios/diff_ddim50_ep3000 --obs-output results/diffusion_obs.csv

# Tables and figures from the per-observation CSV:
python eval_hedging.py --obs results/diffusion_obs.csv --figdir results/figures \
  --alpha-dir results/alpha_robustness

# Data-descriptive figures 1 and 2:
python eval_data_figures.py --figdir results/figures
```

Committed results (`results/`) let you read Table 2 and the figures without rerunning.
Full method, design justifications, and the result tables are in `CLAUDE.md`.

## Open problems

- The committed numbers use the fast DDIM-50 sampler. Headline/publishable numbers
  must be re-run with `--sampling-timesteps 200` (faithful DDPM) into a separate cache.
- Delta hedging performance seems orthogonal to diffusion-driven hedging. This shouldn't be the case and we need to debug why this happens. 
- The BS-vs-C/S consistency MAE differs from the PhD's vol-surface notebook by a fixed
  scale (a units/convention difference, not a model issue) — treat it as relative until
  resolved.
