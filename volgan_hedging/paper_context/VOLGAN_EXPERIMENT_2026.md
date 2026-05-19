# VolGAN Experiment Log 2026

Project path: `/home/yinbinha/VolGAN`.

## 2026-05-18 run state

Goal: train shared-grid VolGAN, tune toward VolGAN paper-style diagnostics, then feed generated one-day scenarios into the data-driven hedging layer and compare against delta / delta-vega benchmarks.

Code changes:
- Added `volgan_experiment.py` as an experiment driver for shared-grid training, checkpointing, diagnostics, VolGAN scenario export, and hedging comparison.
- The driver uses the existing `VolGAN.Generator` / `Discriminator` and `hedging.py` APIs, but implements checked finite-difference smoothness matrices for the 11x9 m-major/tau-minor shared grid.
- It adds SPX as an experiment-layer hedge instrument and can trim option hedge candidates by observed quote coverage, because the raw OptionMetrics option data is filtered to positive-volume quote days.

Data/preprocessing:
- Full shared-grid preprocessing completed at `data/processed_shared_grid_11x9`.
- Accepted dates: 5,825 from 2000-01-04 through 2023-02-28; no drops in `audit_manifest.json`.
- VolGAN one-step samples after 22-day lag: 5,803, with condition dim 102 and output dim 100.

Verification so far:
- Baseline checks passed: `python3 -B -m py_compile VolGAN.py hedging.py shared_grid_preprocessing.py`; `python3 hedging.py --solver-self-check`; `python3 hedging.py --scenario-adapter-self-check`; `python3 hedging.py --backtest-self-check`; `python3 hedging.py --paper-output-self-check`.
- Experiment driver self-check passed: `CUDA_VISIBLE_DEVICES=3 python3 volgan_experiment.py --stage self-check --device cuda:0 --output-dir results/volgan_selfcheck_20260518`.
- Lite tuning run completed at `results/volgan_tune_lite_20260518` for `paper_lite` and `wide_lite`; `paper_lite` ranked first by the current selection score.

Active long jobs when this note was first written:
- Full paper-hyperparameter training: `CUDA_VISIBLE_DEVICES=3 python3 volgan_experiment.py --stage train --device cuda:0 --configs paper --output-dir results/volgan_paper_20260518`, log `results/logs/volgan_paper_20260518.log`, process observed as PID 3554490. Hyperparameters: noise 32, hidden 16, gradient matching 25, epochs 10000, batch 100, train split ending 2018-06-16.
- Lite-checkpoint broad hedging validation: `python3 volgan_experiment.py --stage hedge --device cpu --checkpoint results/volgan_tune_lite_20260518/paper_lite.pt --hedge-samples 256 --hedge-max-periods 6 --hedge-m0 0.75,0.8,0.9,1.1,1.2,1.25 --min-hedge-observed-frac 0.5 --output-dir results/volgan_hedge_lite_6period_20260518`, log `results/logs/volgan_hedge_lite_6period_20260518.log`, process observed as PID 3557624.

Important caveats:
- The current shared-grid experiment is hedging-compatible and follows the final paper-style 11x9 grid. It is not an exact reproduction of older 10x8 VolGAN paper tables.
- Hedging realized rows are sparse under strict observed-quote mechanics because selected option contracts are absent on many positive-volume-filtered days. The experiment driver records skipped intervals and retained hedge counts; claims must be framed as liquidity-filtered unless a separate fill/interpolation policy is approved and implemented.
- The current comparison uses raw VolGAN scenarios, not reweighted scenarios, matching the hedging-paper preference for raw generator samples.
