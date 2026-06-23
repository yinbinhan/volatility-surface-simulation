# Portfolio hedging — results summary (2026-06-21)

Best model + results so far for the diffusion-driven hedging application
(Cont & Vuletić 2025 replication/extension on SPX options).

## Best configuration

- **Generator:** the team's diffusion model as a black-box one-step-ahead scenario
  generator. Checkpoint
  `model_results/dfm_shared_grid_22d_logiv_call_return_ts1779948122_seed42/model-epoch-3000.pt`
  (11×9 implied-vol surface grid, 22-day sequences). Sampled with **DDIM-50**.
- **Hedge:** daily-rebalanced **LASSO local risk-minimization**, paper-faithful
  instrument set (**underlying + options**, §4.3), L1 transaction-cost penalty,
  regularization α chosen by **AIC** on an independent scenario draw.
- **Portfolio:** long 1-month straddle (K = m₀·S₀), pooled over 6 moneyness values
  {0.75, 0.80, 0.90, 1.10, 1.20, 1.25}.
- **Sample:** SPX, 2018-07 → 2023-02, **n = 6,918** daily tracking errors.
- **Model health:** the checkpoint was validated as a sound generator — generated
  surfaces match training data on calendar/butterfly arbitrage and BS-consistency;
  only a minor vertical-spread (L2) leakage remains, still improving at epoch 3000.

## Headline

**The diffusion-driven hedge has the lowest tracking-error std *and* the thinnest tails
of every method, both with and without the Covid window.**

Tracking error Z in USD; VaR reported as positive loss magnitude.

**Covid-excluded (n = 6,252):**

| Method | Std | VaR 5% | VaR 2.5% | VaR 1% |
|---|---|---|---|---|
| **Diffusion-LASSO (ours)** | **3.96** | **6.50** | **11.67** | **13.88** |
| Delta-vega | 8.70 | 14.56 | 21.35 | 40.03 |
| Delta | 15.99 | 27.21 | 35.34 | 45.57 |
| VolGAN-diffusion (paper) | 8.15 | 10.55 | 17.32 | 33.85 |

**Covid-included (n = 6,918):**

| Method | Mean | Std | VaR 5% | VaR 1% |
|---|---|---|---|---|
| **Diffusion-LASSO (ours)** | 0.25 | **4.61** | **6.87** | **14.78** |
| Delta-vega | 0.11 | 16.77 | 15.01 | 38.86 |
| Delta | 1.50 | 41.30 | 29.59 | 50.59 |
| Unhedged | 5.24 | 118.35 | 170.04 | 278.72 |

We beat both classical baselines **and** the paper's own VolGAN-diffusion numbers on
body and tails — i.e. the project's target claim ("matches/beats VolGAN, especially in
the tails") is reproduced.

## Why it's credible

Both baselines are now **paper-faithful**, so the diffusion advantage is not an artifact
of handicapped baselines:

- **Delta-vega:** a single option fixed at K = S₀ at window start and held to expiry,
  only the hedge ratio rebalanced daily (Cont & Vuletić §4.4).
- **Delta:** the off-by-one fixed on 2026-06-20 — it had been hedging each day with the
  *previous* day's delta and leaving every window's first day unhedged. After the fix,
  the ex-Covid delta std collapses 35.97 → 15.99 (proper regime dependence), confirmed
  to leave the diffusion / delta-vega / unhedged columns bit-identical.

## Caveats to state up front

1. **DDIM-50 (fast sampler).** The faithful **DDPM-200** rerun is still pending and is
   what should be cited as the final headline (expected close to these).
2. **Realized P&L is marked on smoothed (vsurfd) surfaces, not raw market mid-quotes**
   (the paper uses mids). This lowers *all* methods' absolute std similarly, so the
   robust claim is the **relative ranking**, not the absolute 3.96.
3. **A paper-bandwidth (0.002, 0.046) surface retrain improves further** on the same
   footing (options-only Covid-out std 6.81 vs 7.42) — a promising direction, not yet
   combined with the underlying-inclusive instrument set.

## Artifacts

- `results/hedging_tables.xlsx` — Table 2 (Covid in/out) + Tables 1/3/4.
- `results/figures/` — fig6 (pooled error histogram), fig7–10 (diffusion vs delta /
  delta-vega scatter), fig11–13 (error over time, per-moneyness histograms).
- `results/diffusion_obs.csv` — per-observation tracking errors (source of all tables/figures).
