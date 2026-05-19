# Data-Driven Hedging 2025 Paper Context

Paper: Rama Cont and Milena Vuletic, *Data-driven hedging with generative models*, dated 2025-06-06 in the local PDF.

Local PDF reference used for this context: `/Users/yinbinhan/Downloads/5282525.pdf`.

## Role in this project

This paper defines the target hedging experiment. The current repo does not implement the hedging layer. Our goal is to reproduce the paper's hedging setup and then replace VolGAN scenarios with diffusion-model scenarios while keeping the hedging instruments, transaction-cost treatment, and tracking-error evaluation comparable.

## Core methodology

At each rebalance date, a conditional scenario generator produces one-step-ahead scenarios for the underlying and risk factors. Given simulated changes in the target portfolio and hedging instruments, hedge ratios are computed by local conditional risk minimization.

With transaction costs, the paper solves an L1-regularized regression:

- response: simulated target portfolio value changes;
- design matrix: simulated hedging-instrument value changes;
- intercept: estimated conditional mean term;
- penalty: proportional to `alpha * g0 * c_i * |phi_i - phi_{i,prev}|`;
- `c_i` is one-half of the bid-ask spread for instrument `i`.

The appendix gives coordinate descent with soft-thresholding. This layer must be implemented separately.

## Hedging experiment

Target portfolio:

- one-month long straddle;
- initial strike `K = m0 * S0`;
- paper uses `m0` values `{0.75, 0.8, 0.9, 1.1, 1.2, 1.25}`.

Hedging frequency:

- daily, with `dt = 1/252`;
- non-overlapping one-month periods;
- new long straddle entered after expiry.

Candidate hedging set:

- underlying SPX;
- one-month calls and puts with initial moneyness `{0.9, 0.95, 0.975, 1, 1.025, 1.05, 1.1}`;
- `m < 1` are puts, `m >= 1` are calls;
- exclude options that are part of the target straddle.

Benchmarks:

- delta hedging with the underlying;
- delta-vega hedging with the underlying and an initially ATM option;
- data-driven scenario regression hedge.

## Regularization selection

The paper selects `alpha` by validation on independent generator samples:

- fit hedge coefficients on one set of generated scenarios;
- compute an AIC-style criterion on an independent scenario set;
- search over `A = {0.01, 0.02, ..., 0.2}` in the 2025 paper;
- robustness checks consider wider grids such as `{0.01, 0.05, 0.1, 0.5, 1}`.

For our diffusion comparison, use the same candidate alpha grid unless explicitly testing sensitivity.

## Required data from OptionMetrics

The downloaded dataset is designed for this task:

`data/optionmetrics_spx_20000103_20230228/`

Use:

- `underlying/` for SPX close prices and returns;
- `raw_options/` for option quotes, IVs, deltas, vegas, maturities, strikes, and half-spreads;
- `vol_surface_delta_grid/` only as a diagnostic or alternate source of pre-smoothed IVs.

Columns already present in `raw_options`:

- identifiers and dates: `secid`, `date`, `symbol`, `exdate`, `optionid`;
- option descriptors: `cp_flag`, `strike_price`, `strike`, `days_to_exp`, `ttm`, `moneyness`;
- market quotes: `best_bid`, `best_offer`, `mid_price`, `bid_ask_spread`, `half_spread`, `volume`, `open_interest`;
- greeks/IV: `impl_volatility`, `delta`, `gamma`, `vega`, `theta`;
- underlying alignment: `spot`.

## Data-processing requirements

For each one-month experiment:

- identify the start date and expiry date;
- construct the target straddle using nearest available contracts to `K = m0 S0` and one-month expiry;
- construct candidate hedging instruments with initial moneyness near the specified grid;
- track daily mid-price values, deltas, vegas, and half-spreads for the selected contracts through expiry;
- build smoothed IV and bid/ask-spread surfaces on the same `(m, tau)` grid used by the scenario generator;
- keep raw contract identifiers so realized hedging PnL can be audited.

## Evaluation outputs

Reproduce at least:

- tracking error time series for each method and `m0`;
- pooled tracking-error statistics: mean, median, variance, standard deviation, VaR at 5%, 2.5%, and 1%;
- comparison with and without the Covid-19 stress interval if following the paper;
- number of selected hedging instruments over time;
- delta and vega exposure of the hedged position.

## Diffusion-model comparison rule

When replacing VolGAN with a diffusion model, keep fixed:

- same market data;
- same target straddles;
- same candidate hedging instruments;
- same transaction-cost proxy;
- same alpha-selection rule;
- same tracking-error metrics.

Only the conditional scenario generator should change unless the experiment explicitly states otherwise.
