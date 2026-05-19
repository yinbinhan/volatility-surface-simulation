# VolGAN 2024 Paper Context

Paper: Milena Vuletic and Rama Cont, *VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces*, Applied Mathematical Finance 31(4), 203-238.

Local PDF reference used for this context: `/Users/yinbinhan/Downloads/VolGAN  A Generative Model for Arbitrage-Free Implied Volatility Surfaces.pdf`.

## Role in this project

This paper is the baseline scenario-generator target. First reproduce the VolGAN data pipeline and generator behavior on SPX implied-volatility surfaces. Then use the same state variables and evaluation conventions when developing the diffusion scenario generator.

## State variables and output

VolGAN models the one-step joint dynamics of:

- SPX underlying log-return.
- SPX implied volatility surface on a fixed moneyness/time-to-maturity grid.

The model conditions on recent state information and outputs:

- one simulated underlying log-return;
- one simulated increment of the log implied-volatility surface.

In code this appears as an output vector of length 81: component 0 is return, components 1:81 are flattened log-IV increments.

## Static arbitrage conventions

The implied-volatility surface is represented by moneyness `m = K / S_t` and time to maturity `tau = T - t`. The paper measures static-arbitrage violations through relative call prices on the grid. The three penalty components are:

- calendar: call price increasing in maturity;
- call monotonicity: call price decreasing in moneyness/strike;
- butterfly: call price convex in moneyness/strike.

The code implements these as `penalty_mutau`, `arbitrage_penalty`, and tensor variants.

## VolGAN architecture and loss

VolGAN is a customized conditional GAN. The important reproduction details are:

- generator and discriminator are simple feedforward networks in this repo;
- generated surfaces are regularized by discrete Sobolev smoothness penalties in moneyness and maturity;
- smoothness weights are calibrated by gradient-norm matching;
- the paper first runs `n_grad = 25` BCE-only epochs to estimate gradient ratios;
- the main training then restarts from the same initialization and trains with BCE plus smoothness penalties;
- the upstream example uses `n_epochs = 10000`.

Scenario reweighting is a post-hoc correction using the arbitrage penalty. The paper reports that reweighting can reduce arbitrage-penalty tails, but later hedging experiments may use raw generator output when raw scenarios mimic the market better.

## Data and grid

The paper trains on SPX OptionMetrics option-price data from January 2000 through 2023-02-28. The extracted code uses:

- moneyness grid: `np.linspace(0.6, 1.4, 10)`;
- maturity grid: 8 selected day-to-expiry points from the OptionMetrics data;
- flattened surfaces of length 80;
- 21-day realized volatility and shifted return features in `DataPreprocesssing`.

The paper smooths OptionMetrics option prices into IV surfaces using a kernel method based on Cont and da Fonseca. The code has both ordinary and vega-weighted Nadaraya-Watson smoothing helpers in `datacleaning.py`.

## Baseline evaluation targets

For VolGAN reproduction, prioritize:

- reconstruct `surfaces_transform.csv` on the `(m, tau)` grid;
- train VolGAN using the upstream defaults or documented changes;
- compare arbitrage penalties of data, BCE GAN, raw VolGAN, and reweighted VolGAN;
- inspect one-step ATM and OTM implied-volatility forecasts;
- compare SPX return simulations;
- compute PCA of log-IV increments and compare data vs simulations;
- optionally compute VIX from simulated surfaces with `VIX(...)`.

## Data available here

Use `data/optionmetrics_spx_20000103_20230228/`:

- `raw_options/` for raw option-level rows used to reconstruct smoothed IV surfaces and bid/ask surfaces;
- `vol_surface_delta_grid/` for OptionMetrics pre-smoothed delta-grid surfaces, useful as a diagnostic or alternative starting point;
- `underlying/` for reproducible SPX close prices instead of Yahoo/SPY.

## Open implementation tasks

- Build a local, reproducible preprocessing script that creates the VolGAN `data.csv` and `surfaces_transform.csv` equivalents from the downloaded WRDS data.
- Decide whether to reproduce the paper's smoothing exactly from raw option prices or use OptionMetrics pre-smoothed delta-grid rows as an intermediate.
- Replace Yahoo/SPY price pulls in `SPXData(...)` with the downloaded SPX `secprd` close prices.
- Record every preprocessing choice because the hedging task depends on consistent surfaces, option prices, spreads, and greeks.
