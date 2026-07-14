# Diffusion-Generated Volatility Surfaces for Option Hedging

Research code for conditional simulation of daily SPX implied-volatility surfaces and their use in sparse, transaction-cost-aware option hedging.

The pipeline:

1. smooth licensed OptionMetrics quotes onto a common surface grid;
2. construct rolling sequences of implied-volatility surfaces and SPX returns;
3. train the Adapted Sequential Diffusion model;
4. fine-tune low-rank adapters with an implementation-level static-arbitrage penalty; and
5. use generated one-day-ahead scenarios to construct option hedges and compare them with unhedged, delta, and delta-vega strategies.

> **Publication status.** Source code is public. Licensed OptionMetrics data, trained checkpoints, and bulk experiment outputs are not distributed. Existing development checkpoints were created from working trees recorded as dirty in their metadata and are therefore not presented as archival release artifacts.

## Method overview

The Adapted Sequential Diffusion model conditions on 21 trading days and generates the next day's SPX log return jointly with its log-implied-volatility surface. A causal transformer parameterizes the score network in a sequential Gaussian diffusion process. The surface is represented on an 11 × 9 moneyness-maturity grid.

LoRA fine-tuning penalizes calendar, vertical-spread, and butterfly violations after reconstructing normalized call prices from generated implied volatilities. This penalty is a numerical diagnostic and training objective; it is not a certificate that every sampled surface is arbitrage-free.

For hedging, diffusion scenarios enter a transaction-cost-regularized LASSO problem. The eligible instruments are the underlying and a panel of options. The same local accounting implementation evaluates the diffusion hedge and the classical baselines.

This implementation builds on [Diffusion Factor Models](https://arxiv.org/abs/2504.06566). The hedging design follows [Cont and Vuletić (2025)](https://doi.org/10.1007/s10479-025-06867-3).

~~~text
licensed OptionMetrics quotes
          |
          v
daily 11 x 9 volatility surfaces
          |
          v
rolling tensors (N, S, C, H, W)
          |
          v
adapted sequential diffusion
          |
          +--> LoRA fine-tuning with arbitrage penalties
          |
          v
one-day-ahead return and surface scenarios
          |
          v
transaction-cost-aware sparse hedging
          |
          v
tables, diagnostics, and figures
~~~

## Reference experiment

The values below record the current development configuration.

| Item | Reference value |
|---|---|
| Shared grid | 11 moneyness points × 9 maturities |
| Tensor layout | `(N, S, C, H, W)` with `H = maturity` and `W = moneyness` |
| Training sample | 4,621 rolling windows of shape `(22, 2, 9, 11)` |
| Channels | log implied volatility; broadcast SPX log return |
| Conditioning | 21 observed days; generate day 22 |
| Training cutoff | target date no later than 2018-06-16 |
| Diffusion network | transformer width 256, 8 layers, 4 heads |
| Diffusion process | 200 steps, cosine schedule, `pred_x0` objective |
| Base training | 1,000 epochs, seed 20260623 |
| LoRA fine-tuning | rank 8, 200 steps, batch 32, KL weight 0.1 |
| Hedging period | 2018-07-01 through 2023-02-28 |
| Target moneyness | 0.75, 0.80, 0.90, 1.10, 1.20, 1.25 |
| Scenario counts | diffusion: 100; validation: 100 |
| COVID reporting | both included and excluded |

## Current empirical interpretation

The following statements describe the fixed development run. They are implementation results, not formal statistical conclusions.

- The COVID-included and COVID-excluded evaluations use 56 and 51 matched windows, respectively, for the base and fine-tuned diffusion checkpoints.
- Base and fine-tuned diffusion have lower pooled tracking-error standard deviation than delta hedging in both evaluations.
- Delta-vega has a larger 1% lower-tail loss than both diffusion checkpoints in both evaluations.
- Fine-tuning substantially lowers the measured aggregate static-arbitrage penalty, but it does not lower pooled tracking-error standard deviation in either evaluation; the tail results are mixed.

No confidence intervals or hypothesis tests are reported. Absolute values should not be described as a numerical reproduction of prior work because the original hedging implementation is not public and this repository provides a local reimplementation.

## Repository map

| Path | Purpose |
|---|---|
| `shared_grid_preprocessing.py` | Convert raw SPX option quotes into daily 11 × 9 surfaces |
| `prepare_shared_grid_data.py` | Convert daily surfaces into rolling diffusion tensors |
| `adapted_sequential_diffusion/` | Adapted sequential diffusion implementation |
| `config/config.py` | Model, optimization, and sampling defaults |
| `train.py` | Base training, conditional sampling, and run metadata |
| `fine_tune.py` | Online LoRA fine-tuning |
| `fold_lora.py` | Merge LoRA weights into a plain diffusion checkpoint |
| `sample_lora.py` | Sample directly from a LoRA checkpoint |
| `hedging.py` | Instrument panels, LASSO solver, scenario adapters, and self-checks |
| `hedging_backtest_utils.py` | Model-neutral pricing and backtest utilities |
| `delta_surface.py` | Delta-grid interpolation and Black-Scholes marking |
| `implied_rate.py` | Put-call-parity net-carry estimator |
| `backtest_diffusion.py` | Diffusion, unhedged, delta, and delta-vega evaluation |
| `plot_surface_data.py` | Market-surface spread and arbitrage-diagnostic figures |
| `summarize_hedging_results.py` | Aggregate hedging metrics and tables |
| `plot_hedging_figures.py` | Hedging figures |
| `sanity_check.py` | Window-set and result diagnostics |
| `build_CD.py` | Build selected tables and figures from a local result bundle |

## Installation

The project has been exercised with Python 3.10. Install a PyTorch build compatible with the local CUDA runtime before installing the remaining dependencies.

~~~bash
git clone https://github.com/yinbinhan/volatility-surface-simulation.git
cd volatility-surface-simulation
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

`requirements.txt` lists the direct dependencies of the retained source tree. It is not a fully pinned CUDA/PyTorch lock.

### Data-free verification

These deterministic checks do not require OptionMetrics data or model checkpoints:

~~~bash
python implied_rate.py --self-check
python hedging.py --solver-self-check
python hedging.py --scenario-adapter-self-check
python hedging.py --backtest-self-check
python hedging.py --paper-output-self-check
~~~

## Data

The raw SPX option data are proprietary OptionMetrics data and are not distributed by this repository. Users must obtain access under their own license. Generated tensors, checkpoints, and full experiment outputs are also excluded from Git.

Expected input layout:

~~~text
data/optionmetrics_spx_20000103_20230228/
  raw_options/
    spx_options_YYYY.csv.gz
  underlying/
    spx_secprd_YYYY.csv.gz
  vol_surface_delta_grid/
    spx_vsurfd_YYYY.csv.gz
~~~

The preprocessing path uses dates, expiries, call/put flags, strikes, bid/offer quotes, implied volatility, volume, and, when available, vega. Underlying files must provide daily SPX closes. Delta-grid surface files are used for realized P&L marking in the hedging backtest.

The shared grid uses moneyness

~~~text
0.60, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.40
~~~

and maturity in years

~~~text
1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1
~~~

## Diffusion workflow

### 1. Build daily surfaces

~~~bash
python shared_grid_preprocessing.py \
  --data-root data/optionmetrics_spx_20000103_20230228 \
  --output-dir data/processed_shared_grid_11x9 \
  --self-check
~~~

The output includes `grid_config.json`, `surface_tensor.npz`, compressed surface tables, and `audit_manifest.json`.

### 2. Build 22-day rolling windows

~~~bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_iv_22 \
  --channel-mode iv \
  --seq-len 22 \
  --conditioning-length 21 \
  --self-check
~~~

The resulting tensor has layout `(N, 22, 2, 9, 11)`. Its channels are `log_iv` and `log_return_broadcast`. Output filenames retain the historical `shared_grid_30d_...` stem; the metadata records the actual sequence length and conditioning target.

### 3. Train the base model

~~~bash
python train.py \
  --data_path data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --conditioning_path data/shared_grid_iv_22/shared_grid_30d_conditioning.npy \
  --conditioning_length 21 \
  --window_length 22 \
  --epochs 1000 \
  --seed 20260623 \
  --gpu 0
~~~

Each run writes a separate directory under `model_results/` and records its commit, dirty-tree state, arguments, and configuration snapshot.

### 4. Fine-tune with the IV-based arbitrage reward

~~~bash
python fine_tune.py \
  --checkpoint model_results/BASE_RUN/model-epoch-1000.pt \
  --data_path data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --reward_mode iv_bs \
  --steps 200 \
  --batch_size 32 \
  --lr 1e-4 \
  --kl_weight 0.1 \
  --lora_rank 8 \
  --transition_chunk_size 50 \
  --grad_checkpoint 1 \
  --save_path ft_lora_final.pt \
  --save_every 25
~~~

For the two-channel data, `iv_bs` reconstructs normalized call prices from channel-0 log implied volatility before evaluating calendar, vertical-spread, and butterfly penalties.

The backtest loader expects a plain diffusion state dictionary. Fold a selected LoRA checkpoint before using it:

~~~bash
python fold_lora.py \
  --in model_results/fine_tuning/FT_RUN/model-step-200.pt \
  --out model_results/fine_tuning/FT_RUN/model-step-200-folded.pt \
  --rank 8 \
  --alpha 16
~~~

Direct sampling from a LoRA checkpoint is available through `sample_lora.py`.

## Hedging workflow

### 1. Estimate daily net carry

~~~bash
python implied_rate.py \
  --data-root data/optionmetrics_spx_20000103_20230228 \
  --start-year 2017 \
  --end-year 2022 \
  --out data/implied_rate.csv \
  --self-check
~~~

The backtest uses this daily put-call-parity estimate by default. Use `--risk-free-mode zero` only for an intentional zero-carry comparison.

### 2. Run diffusion hedging

~~~bash
python backtest_diffusion.py \
  --checkpoint model_results/BASE_RUN/model-epoch-1000.pt \
  --train-data data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --processed-dir data/processed_shared_grid_11x9 \
  --prepared-dir data/processed_shared_grid_11x9 \
  --data-dir data/optionmetrics_spx_20000103_20230228 \
  --m0 1.1 \
  --n-scenarios 100 \
  --n-val 100 \
  --rate-table data/implied_rate.csv \
  --output results/diffusion_m1p1.csv
~~~

Repeat the command with a folded fine-tuned checkpoint for the fine-tuned evaluation. Use `--all-m0` to pool the six reference moneyness values and `--exclude-covid` for the COVID-excluded evaluation.

The driver writes a summary CSV and, when method output lengths agree, a companion `_raw.csv` file with interval-level tracking errors and metadata.

## Results and reproducibility boundaries

`sanity_check.py` audits window counts, alignment, COVID coverage, and descriptive statistics. `build_CD.py`, `summarize_hedging_results.py`, and `plot_hedging_figures.py` consume local result bundles to construct tables and figures.

Result directories are excluded from Git. A publication artifact should archive the clean source commit, preprocessing and training metadata, base and folded fine-tuned checkpoints, raw backtest CSVs, sanity reports, and generated exhibits.

Important boundaries:

- **Licensed data:** OptionMetrics files cannot be redistributed here.
- **Current artifacts:** development checkpoints and result bundles are local-only.
- **Protocol scope:** the hedging code is a local reimplementation because the original implementation is not public.
- **Inference:** reported comparisons are descriptive; no confidence intervals or hypothesis tests establish universal superiority.
- **Arbitrage diagnostic:** the implemented penalty measures sampled-surface violations but does not certify arbitrage freedom.

## References

- Minshuo Chen, Renyuan Xu, Yumin Xu, and Ruixun Zhang. [*Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure*](https://arxiv.org/abs/2504.06566), 2025. [Upstream implementation](https://github.com/xuym/diffusion-factor-model).
- Rama Cont and Milena Vuletić. [*Data-driven hedging with generative models*](https://doi.org/10.1007/s10479-025-06867-3), *Annals of Operations Research*, 2025.

When citing this software, record the exact repository commit together with the methodological sources above.

## License

The repository is released under the [MIT License](LICENSE). The license applies to this repository's code, not to licensed OptionMetrics data.
