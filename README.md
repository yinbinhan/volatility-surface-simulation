# 🌊 Diffusion Factor Model

<p align="center">
  <img src="assets/demo.png" alt="Diffusion Factor Model Demo" width="700"/>
</p>

This repository implements a Diffusion Factor Model for financial data.

## 📝 Summary

Diffusion Factor Model (DFM) is a novel approach that adapts diffusion models to generate new financial returns with realistic factor structure. It achieves superior performance in preserving the statistical properties and latent factor patterns of financial data, making it valuable for portfolio optimization and risk management applications.

## ✨ Features

- 📊 Diffusion models adapted for financial data with factor structure
- 🔄 Support for both simulation data (num_samples, height, width) and empirical data (num_samples, length) formats
- 📐 Automatic adaptation to different input dimensions
- 💹 Portfolio optimization evaluation framework
- 📈 Factor recovery evaluation metrics

## 🔧 Installation

```bash
git clone https://github.com/xymmmm00/diffusion_factor_model.git
cd diffusion-factor-model
pip install -r requirements.txt
```

For portfolio optimization, MOSEK requires a license (free for academic use).

## 📁 Project Structure

```
diffusion-factor-model/
├── config/                      # Configuration settings
├── diffusion_factor_model/      # Core model implementation
├── eval/                        # Evaluation modules
├── simulation_experiment_data/  # Simulation data storage
├── empirical_analysis_data/     # Empirical data storage
├── model_results/               # Trained models (created automatically)
├── samples/                     # Generated samples (created automatically)
└── train.py                     # Main training script
```

## 🚀 Training

The training script automatically detects data format and adapts the model architecture accordingly.

```bash
# Train with simulation data:
python train.py --data_path /path/to/simulation_experiment_data/training_data_example.npy --seed 42 --gpu 0

# Train with empirical data:
python train.py --data_path /path/to/empirical_analysis_data/training_data_example.npy --seed 42 --gpu 0
```

## OptionMetrics Volatility-Surface Hedging Workflow

For the SPX volatility-surface hedging experiments, start from raw OptionMetrics files and build the shared-grid dataset used by diffusion and the hedging layer.

Expected raw-data layout:

```text
data/optionmetrics_spx_20000103_20230228/
  raw_options/spx_options_YYYY.csv.gz
  underlying/spx_secprd_YYYY.csv.gz
```

Build daily 11x9 shared-grid surfaces:

```bash
python shared_grid_preprocessing.py \
  --data-root data/optionmetrics_spx_20000103_20230228 \
  --output-dir data/processed_shared_grid_11x9 \
  --self-check
```

Build matched IV-only diffusion windows, with 21 observed trading days and the next day as the generated target:

```bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_iv_22 \
  --channel-mode iv \
  --seq-len 22 \
  --conditioning-length 21 \
  --self-check
```

Train and sample diffusion:

```bash
python train.py \
  --data_path data/shared_grid_iv_22/shared_grid_30d_logiv_return.npy \
  --conditioning_path data/shared_grid_iv_22/shared_grid_30d_conditioning.npy \
  --conditioning_length 21 \
  --gpu 0
```

Generated samples are written under `samples/dfm_*`. The first `conditioning_length` days are fixed from the observed prefix, and the next index is generated. These samples can be converted into hedging scenarios through `hedging.py` using `IVSurfaceScenarios`.

For the paper-style call-price setup, use:

```bash
python prepare_shared_grid_data.py \
  --processed-dir data/processed_shared_grid_11x9 \
  --output-dir data/shared_grid_call_30 \
  --channel-mode paper \
  --seq-len 30 \
  --conditioning-length 29 \
  --self-check
```

Run hedging component checks:

```bash
python hedging.py --solver-self-check
python hedging.py --scenario-adapter-self-check
python hedging.py --backtest-self-check
python hedging.py --paper-output-self-check
```

See `SHARED_GRID_HEDGING_WORKFLOW.md` for the concise end-to-end checklist. Full hedging performance evaluation still requires trained checkpoints plus an exporter that emits `K` one-day-ahead scenarios per hedge date.

### Supported Data Formats

1. **Empirical data**: Shape `(samples, assets)` - e.g., `(1024, 512)` 
2. **Simulation data**: Shape `(samples, height, width)` - e.g., `(512, 32, 64)`

## 📊 Evaluation

The repository includes evaluation modules for:

1. **Mean and Covariance Calculation** - With winsorization and shrinkage estimation
2. **Simulation Evaluation** - Comparing generated distributions (both return and latent subspace) with ground truth
3. **Mean-Variance Portfolio Evaluation** - Creating mean-variance portfolios with performance metrics
4. **Factor Timing Portfolio Evaluation** - Using PCA, POET, RP-PCA for factor-based portfolios

<p align="center">
  <img src="assets/distribution_example.png">
</p>

<p align="center">
  <img src="assets/portfolio_example.png">
</p>

## 📚 Citation

```
@article{chen2025diffusion,
  title={Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure},
  author={Chen, Minshuo and Xu, Renyuan and Xu, Yumin and Zhang, Ruixun},
  journal={arXiv preprint arXiv:2504.06566},
  year={2025}
}
```
