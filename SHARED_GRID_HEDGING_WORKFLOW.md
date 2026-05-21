# Shared-Grid Diffusion and Hedging Workflow

This repo can now build the common OptionMetrics surface dataset, train diffusion on it, and run hedging plumbing from generated scenarios.

## Expected Data Layout

Place the OptionMetrics/SPX data under:

```text
data/optionmetrics_spx_20000103_20230228/
  raw_options/spx_options_YYYY.csv.gz
  underlying/spx_secprd_YYYY.csv.gz
```

## 1. Build Shared Daily Surfaces

```bash
python shared_grid_preprocessing.py \
  --data-root data/optionmetrics_spx_20000103_20230228 \
  --output-dir data/processed_shared_grid_11x9 \
  --self-check
```

This writes `grid_config.json`, `surface_tensor.npz`, daily SPX returns, long IV/price CSVs, and an audit manifest.

## 2. Build Diffusion Windows

Matched IV-only setup:

```bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_iv_22 \
  --channel-mode iv \
  --seq-len 22 \
  --conditioning-length 21 \
  --self-check
```

Paper call-price setup:

```bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_call_30 \
  --channel-mode paper \
  --seq-len 30 \
  --conditioning-length 29 \
  --self-check
```

## 3. Train Diffusion

```bash
python train.py \
  --data_path data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --conditioning_path data/shared_grid_iv_22/shared_grid_30d_conditioning.npy \
  --conditioning_length 21 \
  --gpu 0
```

The generated samples keep the conditioned prefix fixed and generate the next trading day.

## 4. Hedging Smoke Checks

```bash
python hedging.py --solver-self-check
python hedging.py --scenario-adapter-self-check
python hedging.py --backtest-self-check
python hedging.py --paper-output-self-check
```

Current status: tiny VolGAN and diffusion sample outputs have both been converted through `IVSurfaceScenarios` into the LASSO hedging solver arrays. A tiny real-panel LASSO backtest smoke also passed. Full performance evaluation still requires trained model checkpoints plus scenario exporters that generate `K` one-day-ahead scenarios per hedge date.
