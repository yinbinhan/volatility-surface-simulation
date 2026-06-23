# Volatility Surface Simulation and Hedging

This repository contains the diffusion-model side of the SPX volatility-surface
experiments. It builds daily OptionMetrics surfaces on the same fixed grid used
for the VolGAN comparison, trains a sequential diffusion model, samples
one-step-ahead surface scenarios, and provides the hedging utilities needed to
turn generated scenarios into LASSO hedge inputs.

The intended workflow is:

1. Start from raw OptionMetrics SPX option quotes and SPX underlying files.
2. Build a shared 11 x 9 volatility/price surface dataset.
3. Convert the daily surfaces into rolling diffusion windows.
4. Train and sample the diffusion model.
5. Convert generated surfaces into hedging scenarios.

## Repository Layout

```text
.
├── shared_grid_preprocessing.py      # Raw OptionMetrics -> daily 11 x 9 surfaces
├── prepare_shared_grid_data.py       # Daily surfaces -> rolling diffusion windows
├── train.py                          # Sequential diffusion training and sampling
├── hedging.py                        # Instrument panels, scenario adapters, LASSO hedging
├── diffusion_factor_model/           # Sequential diffusion model implementation
├── config/config.py                  # Training and sampling defaults
├── SHARED_GRID_HEDGING_WORKFLOW.md   # Short checklist version of this workflow
└── requirements.txt
```

## Installation

Create a Python environment with PyTorch and the package requirements.

```bash
git clone https://github.com/yinbinhan/volatility-surface-simulation.git
cd volatility-surface-simulation
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build if training on GPU. The code has been smoke
tested on GPU with the shared-grid data path.

## Raw Data Layout

The preprocessing scripts expect the OptionMetrics/SPX files in this layout:

```text
data/optionmetrics_spx_20000103_20230228/
  raw_options/
    spx_options_YYYY.csv.gz
  underlying/
    spx_secprd_YYYY.csv.gz
```

The option files should contain the usual OptionMetrics fields used by
`shared_grid_preprocessing.py` and `hedging.py`, including date, expiry, call/put
flag, strike, bid, offer, implied volatility, delta, vega, volume, and open
interest. The underlying files should contain daily SPX closes.

Large raw data files and generated artifacts should stay outside Git.

## Step 1: Build Shared Daily Surfaces

Run:

```bash
python shared_grid_preprocessing.py \
  --data-root data/optionmetrics_spx_20000103_20230228 \
  --output-dir data/processed_shared_grid_11x9 \
  --self-check
```

This builds the fixed grid:

- Moneyness: `0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4`
- Maturity in years: `1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1`

The output directory contains:

```text
data/processed_shared_grid_11x9/
  grid_config.json
  surface_tensor.npz
  spx_daily.csv.gz
  iv_surfaces.csv.gz
  price_surfaces.csv.gz
  audit_manifest.json
```

`surface_tensor.npz` contains daily tensors for IV, log-IV, normalized call/put
mid prices, normalized half-spreads, SPX close, and log return.

For a quick preprocessing smoke test, use fewer dates:

```bash
python shared_grid_preprocessing.py \
  --data-root data/optionmetrics_spx_20000103_20230228 \
  --output-dir /tmp/shared_grid_smoke \
  --max-dates 35 \
  --self-check
```

## Step 2: Build Diffusion Windows

The diffusion model expects rolling windows with shape:

```text
[num_windows, seq_len, channels, maturity_grid, moneyness_grid]
```

The adapter writes both the training tensor and the conditioning tensor. The
conditioning tensor currently equals the full data tensor; `train.py` uses
`--conditioning_length` to keep only the observed prefix fixed during sampling.

### IV-only matched setup

This is the closest diffusion setup to the VolGAN comparison: use 21 observed
trading days and generate the next trading day.

```bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_iv_22 \
  --channel-mode iv \
  --seq-len 22 \
  --conditioning-length 21 \
  --self-check
```

Channels:

```text
log_iv, log_return_broadcast
```

The main output file is:

```text
data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy
```

### Paper-style call-price setup

This setup includes normalized call prices and uses 29 observed trading days to
generate day 30.

```bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_call_30 \
  --channel-mode paper \
  --seq-len 30 \
  --conditioning-length 29 \
  --self-check
```

Channels:

```text
log_iv, call_mid_over_s, log_return_broadcast
```

The main output file is:

```text
data/shared_grid_call_30/shared_grid_30d_logiv_call_return.npy
```

## Step 3: Train and Sample Diffusion

Train the IV-only matched setup:

```bash
python train.py \
  --data_path data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --conditioning_path data/shared_grid_iv_22/shared_grid_30d_conditioning.npy \
  --conditioning_length 21 \
  --gpu 0 \
  --seed 3407
```

Train the paper-style call-price setup:

```bash
python train.py \
  --data_path data/shared_grid_call_30/shared_grid_30d_logiv_call_return.npy \
  --conditioning_path data/shared_grid_call_30/shared_grid_30d_conditioning.npy \
  --conditioning_length 29 \
  --gpu 0 \
  --seed 3407
```

`train.py` creates one experiment directory under `model_results/` and one under
`samples/`. It records the Git commit, CLI arguments, and config snapshot for
reproducibility.

Typical outputs:

```text
model_results/dfm_<data_id>_ts<timestamp>_seed<seed>/
  model-*.pt
  run_config.json
  commit_hash.txt

samples/dfm_<data_id>_ts<timestamp>_seed<seed>/
  sample_batch1.npy
  sample_batch2.npy
  ...
```

The sample tensors keep the first `conditioning_length` days fixed and generate
the remaining indices. For the one-step experiments above, the generated target
is index 21 in the IV-only setup and index 29 in the paper-style setup.

For a tiny smoke run:

```bash
python train.py \
  --data_path data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --conditioning_path data/shared_grid_iv_22/shared_grid_30d_conditioning.npy \
  --conditioning_length 21 \
  --num_samples 14 \
  --epochs 1 \
  --gpu 0 \
  --seed 9
```

## Step 4: Hedging Utilities

`hedging.py` contains the data-driven hedging layer:

- Builds observed OptionMetrics target-straddle and hedge-instrument panels.
- Solves the transaction-cost LASSO hedge.
- Supports generator-agnostic scenario adapters:
  - direct simulated price changes,
  - selected instrument values,
  - normalized option price surfaces,
  - IV surfaces revalued with Black-Scholes.
- Runs daily backtest mechanics and paper-style summaries.

Run the built-in checks:

```bash
python hedging.py --solver-self-check
python hedging.py --scenario-adapter-self-check
python hedging.py --backtest-self-check
python hedging.py --paper-output-self-check
```

### Paper-compliance review notes

The paper-protocol implementation is concentrated in `hedging.py`:

- `build_instrument_panel(...)` appends one SPX underlying hedge instrument to the option hedge universe, with price `S_t`, delta 1, vega 0, and zero half-spread unless explicitly changed.
- `benchmark_hedge_positions(...)` implements the paper benchmark supports: `delta` uses only the underlying, while `delta_vega` uses the fixed inception-ATM option plus the underlying. There is no least-squares fallback for either benchmark.
- `run_daily_backtest(...)` passes the fixed inception ATM option ID into the benchmark calculation on every rebalance date.
- The scenario adapters partition option contracts from the underlying and use `S_{t+1} - S_t` for the underlying scenario P&L column.
- Daily tracking-error increments use `target_change - hedge_change + transaction_cost`, matching the sign of `Z_t = V_t - Pi_t` when costs reduce tracking-portfolio wealth.
- Paper-mode alpha selection uses the fitted training intercept on the validation batch, counts nonzero fitted positions, and reuses one selected alpha across the window.

The decisive benchmark fixture is in `paper_output_self_check()`: for hedges `[underlying, fixed ATM, distractor]` and target delta/vega `(0.3, 0.4)`, it requires `delta = [0.3, 0.0, 0.0]` and `delta_vega = [-0.5, 2.0, 0.0]`.

Build one observed one-month straddle panel:

```bash
python hedging.py \
  --data-dir data/optionmetrics_spx_20000103_20230228 \
  --start-date 2022-01-21 \
  --m0 1.0 \
  --output-dir data/hedging_panel_2022_01_21_m100
```

The generated model samples still need to be exported into one of the scenario
formats accepted by `hedging.py`. The most relevant routes are:

- IV-only samples -> `IVSurfaceScenarios`
- call-price samples -> `NormalizedPriceSurfaceScenarios`

Both routes produce solver-ready arrays:

```text
target_changes: [num_scenarios]
hedge_changes:  [num_scenarios, num_hedge_instruments]
```

These arrays are the inputs to the LASSO hedging solver.

## Reproducible Experiment Checklist

For every experiment, record:

- Git commit hash.
- Raw data date range.
- Preprocessing command and output directory.
- Diffusion window command and output directory.
- Training command, seed, GPU, and config changes.
- Model checkpoint directory.
- Sample directory.
- Hedging start dates, target moneyness values, and scenario exporter used.

`train.py` records the Git commit and run configuration automatically. The
preprocessing scripts write metadata files in their output directories.

## Current Status

Implemented and smoke tested:

- raw OptionMetrics to shared 11 x 9 surface preprocessing,
- IV-only and paper-style diffusion window adapters,
- conditional diffusion training/sampling path,
- hedging scenario adapters and LASSO checks,
- tiny GPU training/sampling smoke tests,
- tiny real-panel hedging smoke test.

Remaining for full empirical results:

- train production diffusion checkpoints,
- export `K` legal one-day-ahead scenarios per hedge date,
- run the full hedging evaluation over the selected test period,
- compare diffusion against VolGAN using the same data grid and hedging protocol.
