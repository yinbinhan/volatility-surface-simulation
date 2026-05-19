#!/usr/bin/env python3
"""Run shared-grid VolGAN training and hedging experiments.

This script is intentionally outside the upstream VolGAN.py training helper.  It
uses the same model classes and shared-grid loader, but keeps experiment logging,
checkpointing, sample diagnostics, and hedging scenario export in one auditable
place.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datacleaning import interpolate_surface

from VolGAN import (
    DataPreprocesssingSharedGrid,
    Discriminator,
    Generator,
    SharedGridVolGANSmokeCheck,
    arbitrage_penalty,
    penalty_mutau,
    smallBS,
)
from hedging import (
    BacktestSummary,
    DirectScenarioChanges,
    HedgePanel,
    build_instrument_panel,
    greek_residual_summary,
    load_smoothed_surface_market,
    run_daily_backtest,
    selected_hedge_count_turnover_summary,
    skip_summary,
    strategy_comparison_table,
    tracking_error_summary,
    transaction_cost_summary,
)

DEFAULT_ALPHA_GRID = np.round(np.arange(0.01, 0.201, 0.01), 2)
PAPER_HEDGE_M0 = (0.75, 0.8, 0.9, 1.1, 1.2, 1.25)
# Phase 3 / M5 documentation constants -- paper \u00a74.2 specifies N=1000 train
# and M=100 *independent* validation samples per period.  Phase 3 lands the
# protocol enforcement (one alpha freeze per period); Phase 4 / M12 will flip
# the argparse defaults.  These constants are additive and do not change
# current runtime behaviour.
PAPER_N_TRAIN = 1000
PAPER_M_VALIDATION = 100
PAPER_EXPECTED_PERIODS = 52
PAPER_COVID_START = pd.Timestamp("2020-02-13")
PAPER_COVID_END = pd.Timestamp("2020-07-21")
STRATEGY_LABELS = {"lasso": "Data-driven", "delta": "Delta", "delta_vega": "Delta-vega", "unhedged": "Unhedged"}

PAPER_TABLE2_TARGETS = [
    {"sample": "full", "strategy_label": "Unhedged", "mean": 0.16, "median": 0.29, "std": 60.58, "var_5": 78.95, "var_2_5": 102.07, "var_1": 155.29},
    {"sample": "full", "strategy_label": "Delta", "mean": -1.23, "median": -0.44, "std": 32.70, "var_5": 19.49, "var_2_5": 36.33, "var_1": 58.63},
    {"sample": "full", "strategy_label": "Delta-vega", "mean": 0.98, "median": 0.00, "std": 29.70, "var_5": 10.90, "var_2_5": 19.70, "var_1": 43.72},
    {"sample": "full", "strategy_label": "Data-driven", "mean": 0.55, "median": -0.16, "std": 32.98, "var_5": 12.79, "var_2_5": 23.42, "var_1": 50.79},
    {"sample": "covid_excluded", "strategy_label": "Delta", "mean": -1.46, "median": -0.36, "std": 8.40, "var_5": 13.22, "var_2_5": 22.84, "var_1": 36.66},
    {"sample": "covid_excluded", "strategy_label": "Delta-vega", "mean": -0.37, "median": 0.00, "std": 9.34, "var_5": 9.58, "var_2_5": 16.67, "var_1": 34.81},
    {"sample": "covid_excluded", "strategy_label": "Data-driven", "mean": -1.05, "median": -0.18, "std": 8.15, "var_5": 10.55, "var_2_5": 17.32, "var_1": 33.85},
]


@dataclass(frozen=True)
class TrainConfig:
    name: str
    noise_dim: int = 32
    hidden_dim: int = 16
    n_grad: int = 25
    n_epochs: int = 1000
    batch_size: int = 100
    lrg: float = 1e-4
    lrd: float = 1e-4
    seed: int = 20260518
    train_end: str = "2018-06-16"
    eval_samples: int = 256
    eval_max_dates: int = 160


CONFIGS = {
    "smoke": TrainConfig("smoke", noise_dim=4, hidden_dim=8, n_grad=1, n_epochs=1, batch_size=64, eval_samples=16, eval_max_dates=8),
    "paper_lite": TrainConfig("paper_lite", noise_dim=32, hidden_dim=16, n_grad=25, n_epochs=250, batch_size=100, eval_samples=256, eval_max_dates=160),
    "paper_mid": TrainConfig("paper_mid", noise_dim=32, hidden_dim=16, n_grad=25, n_epochs=1000, batch_size=100, eval_samples=512, eval_max_dates=240),
    "paper": TrainConfig("paper", noise_dim=32, hidden_dim=16, n_grad=25, n_epochs=10000, batch_size=100, eval_samples=1000, eval_max_dates=400),
    "wide_lite": TrainConfig("wide_lite", noise_dim=32, hidden_dim=32, n_grad=25, n_epochs=250, batch_size=100, eval_samples=256, eval_max_dates=160),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed_shared_grid_11x9")
    parser.add_argument("--data-dir", default="data/optionmetrics_spx_20000103_20230228")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--configs", default="paper_lite,wide_lite")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=["self-check", "train", "hedge", "report", "end-to-end"], default="end-to-end")
    parser.add_argument("--checkpoint", help="checkpoint path for --stage hedge")
    parser.add_argument("--report-input-dir", help="existing result directory for --stage report")
    parser.add_argument("--hedge-samples", type=int, default=PAPER_N_TRAIN,
                        help="N independent train scenarios per rebalance (paper default 1000).")
    parser.add_argument("--hedge-validation-samples", type=int, default=PAPER_M_VALIDATION,
                        help="M independent validation scenarios per rebalance (paper default 100).")
    parser.add_argument("--hedge-max-periods", type=int, default=52,
                        help="Maximum non-overlapping hedging periods per m0 (paper default 52).")
    parser.add_argument("--hedge-m0", default=",".join(str(x) for x in PAPER_HEDGE_M0))
    parser.add_argument("--hedge-start", default="2018-06-18")
    parser.add_argument("--hedge-schedule", choices=["calendar31", "paper_23td"], default="paper_23td",
                        help="Period schedule (paper default paper_23td = 23 trading days).")
    parser.add_argument("--hedge-end", default="2023-02-28")
    parser.add_argument("--min-hedge-observed-frac", type=float, default=0.0,
                        help="Minimum observed quote fraction (paper default 0.0; use surface quote source).")
    parser.add_argument("--hedge-quote-source", choices=["observed", "surface"], default="surface",
                        help="Quote source (paper default surface: smoothed IV / bid-ask surfaces).")
    parser.add_argument("--hedge-debug-fast", action="store_true",
                        help="Override hedging defaults with the pre-Phase-4 smoke configuration.")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--repricing-vectorize-self-check", action="store_true",
                        help="Run a synthetic correctness + microbenchmark check on the vectorized repricing helpers and exit.")
    args = parser.parse_args()
    if args.hedge_debug_fast:
        _apply_debug_fast_overrides(args)
    return args


def _apply_debug_fast_overrides(args: argparse.Namespace) -> None:
    """Phase 4 / M12: restore the pre-paper smoke configuration when --hedge-debug-fast is set."""
    args.hedge_samples = 512
    args.hedge_validation_samples = 0
    args.hedge_max_periods = 6
    args.hedge_schedule = "calendar31"
    args.min_hedge_observed_frac = 0.85
    args.hedge_quote_source = "observed"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_output_dir(base: str | None) -> Path:
    if base is None:
        base = f"results/volgan_experiment_{utc_stamp()}"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=False)
    return path


def shell_capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - provenance only
        return f"FAILED {cmd}: {exc}"


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(processed_dir: Path) -> dict[str, object]:
    true, condition, m_in, sigma_in, m_out, sigma_out, dates_t, m, tau, ms, taus = DataPreprocesssingSharedGrid(processed_dir)
    true = np.asarray(true, dtype=np.float32)
    condition = np.asarray(condition, dtype=np.float32)
    dates_t = pd.to_datetime(dates_t)
    return {
        "true": true,
        "condition": condition,
        "m_in": np.asarray(m_in, dtype=np.float32),
        "sigma_in": np.asarray(sigma_in, dtype=np.float32),
        "m_out": np.asarray(m_out, dtype=np.float32),
        "sigma_out": np.asarray(sigma_out, dtype=np.float32),
        "dates_t": dates_t,
        "m": np.asarray(m, dtype=float),
        "tau": np.asarray(tau, dtype=float),
        "ms": np.asarray(ms, dtype=float),
        "taus": np.asarray(taus, dtype=float),
    }


def split_indices(dates_t: pd.DatetimeIndex, train_end: str) -> tuple[np.ndarray, np.ndarray]:
    train_mask = dates_t <= pd.Timestamp(train_end)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)
    if len(train_idx) < 32 or len(test_idx) < 2:
        raise ValueError(f"bad split for train_end={train_end}: train={len(train_idx)} test={len(test_idx)}")
    return train_idx, test_idx


def make_diff_matrix(m: np.ndarray, tau: np.ndarray, axis: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lk, lt = len(m), len(tau)
    n = lk * lt
    rows = []
    weights = []
    if axis == "m":
        for i in range(lk - 1):
            step = float(m[i + 1] - m[i])
            for j in range(lt):
                row = torch.zeros(n, dtype=torch.float32, device=device)
                row[i * lt + j] = -1.0
                row[(i + 1) * lt + j] = 1.0
                rows.append(row)
                weights.append(1.0 / max(step * step, 1e-12))
    elif axis == "tau":
        for i in range(lk):
            for j in range(lt - 1):
                step = float(tau[j + 1] - tau[j])
                row = torch.zeros(n, dtype=torch.float32, device=device)
                row[i * lt + j] = -1.0
                row[i * lt + j + 1] = 1.0
                rows.append(row)
                weights.append(1.0 / max(step * step, 1e-12))
    else:
        raise ValueError(axis)
    return torch.stack(rows), torch.tensor(weights, dtype=torch.float32, device=device)


def smoothness_penalty(surface_flat: torch.Tensor, diff: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    diffs = surface_flat @ diff.T
    return torch.mean(torch.sum(weights.reshape(1, -1) * diffs.pow(2), dim=1))


def iter_batches(n: int, batch_size: int, device: torch.device) -> Iterable[torch.Tensor]:
    perm = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        if idx.numel() > 0:
            yield idx


def gradient_matching(
    gen: Generator,
    disc: Discriminator,
    criterion: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    condition_train: torch.Tensor,
    true_train: torch.Tensor,
    m: np.ndarray,
    tau: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    diff_m, weights_m = make_diff_matrix(m, tau, "m", device)
    diff_t, weights_t = make_diff_matrix(m, tau, "tau", device)
    bce_norms: list[float] = []
    m_norms: list[float] = []
    t_norms: list[float] = []
    n_train = condition_train.shape[0]
    for _epoch in range(cfg.n_grad):
        for idx in iter_batches(n_train, cfg.batch_size, device):
            condition = condition_train[idx]
            real = true_train[idx]
            real_and_cond = torch.cat((condition, real), dim=-1)

            disc_opt.zero_grad(set_to_none=True)
            noise = torch.randn((idx.numel(), cfg.noise_dim), device=device)
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)
            disc_loss = 0.5 * (
                criterion(disc(fake_and_cond.detach()), torch.zeros((idx.numel(), 1), device=device))
                + criterion(disc(real_and_cond), torch.ones((idx.numel(), 1), device=device))
            )
            disc_loss.backward()
            disc_opt.step()

            noise = torch.randn((idx.numel(), cfg.noise_dim), device=device)
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)
            fake_iv = torch.exp(condition[:, 3:] + fake[:, 1:])

            for penalty, store in (
                (smoothness_penalty(fake_iv, diff_m, weights_m), m_norms),
                (smoothness_penalty(fake_iv, diff_t, weights_t), t_norms),
                (criterion(disc(fake_and_cond), torch.ones((idx.numel(), 1), device=device)), bce_norms),
            ):
                gen_opt.zero_grad(set_to_none=True)
                penalty.backward(retain_graph=True)
                total = 0.0
                for param in gen.parameters():
                    if param.grad is not None:
                        total += float(param.grad.detach().norm(2).item()) ** 2
                store.append(math.sqrt(total))
            gen_opt.step()
    eps = 1e-12
    alpha = float(np.mean(np.asarray(bce_norms) / np.maximum(np.asarray(m_norms), eps)))
    beta = float(np.mean(np.asarray(bce_norms) / np.maximum(np.asarray(t_norms), eps)))
    diagnostics = {
        "bce_grad_mean": float(np.mean(bce_norms)),
        "m_smooth_grad_mean": float(np.mean(m_norms)),
        "tau_smooth_grad_mean": float(np.mean(t_norms)),
        "alpha": alpha,
        "beta": beta,
    }
    return alpha, beta, diagnostics


def train_main_loop(
    gen: Generator,
    disc: Discriminator,
    criterion: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    condition_train: torch.Tensor,
    true_train: torch.Tensor,
    m: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    beta: float,
    cfg: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    diff_m, weights_m = make_diff_matrix(m, tau, "m", device)
    diff_t, weights_t = make_diff_matrix(m, tau, "tau", device)
    n_train = condition_train.shape[0]
    history = []
    for epoch in range(cfg.n_epochs):
        gen_losses = []
        disc_losses = []
        for idx in iter_batches(n_train, cfg.batch_size, device):
            condition = condition_train[idx]
            real = true_train[idx]
            real_and_cond = torch.cat((condition, real), dim=-1)
            curr = idx.numel()

            disc_opt.zero_grad(set_to_none=True)
            noise = torch.randn((curr, cfg.noise_dim), device=device)
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)
            disc_loss = 0.5 * (
                criterion(disc(fake_and_cond.detach()), torch.zeros((curr, 1), device=device))
                + criterion(disc(real_and_cond), torch.ones((curr, 1), device=device))
            )
            disc_loss.backward()
            disc_opt.step()
            disc_losses.append(float(disc_loss.detach().cpu()))

            gen_opt.zero_grad(set_to_none=True)
            noise = torch.randn((curr, cfg.noise_dim), device=device)
            fake = gen(noise, condition)
            fake_and_cond = torch.cat((condition, fake), dim=-1)
            fake_iv = torch.exp(condition[:, 3:] + fake[:, 1:])
            bce = criterion(disc(fake_and_cond), torch.ones((curr, 1), device=device))
            m_pen = smoothness_penalty(fake_iv, diff_m, weights_m)
            t_pen = smoothness_penalty(fake_iv, diff_t, weights_t)
            gen_loss = bce + alpha * m_pen + beta * t_pen
            gen_loss.backward()
            gen_opt.step()
            gen_losses.append(float(gen_loss.detach().cpu()))
        row = {"epoch": epoch + 1, "gen_loss": float(np.mean(gen_losses)), "disc_loss": float(np.mean(disc_losses))}
        if epoch == 0 or (epoch + 1) % max(1, cfg.n_epochs // 10) == 0 or epoch + 1 == cfg.n_epochs:
            print(f"{cfg.name} epoch={row['epoch']} gen_loss={row['gen_loss']:.6g} disc_loss={row['disc_loss']:.6g}", flush=True)
            history.append(row)
    return history


def build_model(arrays: dict[str, object], cfg: TrainConfig, device: torch.device) -> tuple[Generator, Discriminator]:
    true = arrays["true"]
    condition = arrays["condition"]
    m_in = torch.tensor(arrays["m_in"], dtype=torch.float32, device=device)
    sigma_in = torch.tensor(arrays["sigma_in"], dtype=torch.float32, device=device)
    m_out = torch.tensor(arrays["m_out"], dtype=torch.float32, device=device)
    sigma_out = torch.tensor(arrays["sigma_out"], dtype=torch.float32, device=device)
    gen = Generator(cfg.noise_dim, condition.shape[1], cfg.hidden_dim, true.shape[1], mean_in=m_in, std_in=sigma_in, mean_out=m_out, std_out=sigma_out).to(device)
    disc = Discriminator(condition.shape[1] + true.shape[1], cfg.hidden_dim, mean=torch.cat((m_in, m_out)), std=torch.cat((sigma_in, sigma_out))).to(device)
    return gen, disc


def generate_for_conditions(gen: Generator, conditions: np.ndarray, cfg: TrainConfig, k: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    gen.eval()
    returns = []
    ivs = []
    with torch.no_grad():
        for cond_np in conditions:
            cond = torch.tensor(cond_np, dtype=torch.float32, device=device).reshape(1, -1).repeat(k, 1)
            noise = torch.randn((k, cfg.noise_dim), device=device)
            out = gen(noise, cond).detach().cpu().numpy()
            returns.append(out[:, 0] / math.sqrt(252.0))
            ivs.append(np.exp(cond_np[3:][None, :] + out[:, 1:]))
    return np.stack(returns), np.stack(ivs)


def arbitrage_penalty_batch(iv_flat: np.ndarray, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    lk, lt = len(m), len(tau)
    p_t, p_k, pb_k = penalty_mutau(m, tau * 365.0)
    out = []
    for row in iv_flat.reshape(-1, lk * lt):
        surface = row.reshape(lk, lt)
        calls = smallBS(*np.meshgrid(tau, m), surface, 0) if False else smallBS(np.meshgrid(tau, m)[1], np.meshgrid(tau, m)[0], surface, 0)
        _p1, _p2, _p3, total = arbitrage_penalty(calls, p_t, p_k, pb_k)
        out.append(float(total))
    return np.asarray(out, dtype=float)


def evaluate_model(gen: Generator, arrays: dict[str, object], test_idx: np.ndarray, cfg: TrainConfig, device: torch.device) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed + 17)
    chosen = test_idx.copy()
    if len(chosen) > cfg.eval_max_dates:
        chosen = np.sort(rng.choice(chosen, size=cfg.eval_max_dates, replace=False))
    conditions = arrays["condition"][chosen]
    true = arrays["true"][chosen]
    gen_returns, gen_iv = generate_for_conditions(gen, conditions, cfg, cfg.eval_samples, device)
    data_iv_next = np.exp(conditions[:, 3:] + true[:, 1:])
    data_returns = true[:, 0] / math.sqrt(252.0)
    gen_return_mean = gen_returns.mean(axis=1)
    gen_iv_mean = gen_iv.mean(axis=1)
    data_penalty = arbitrage_penalty_batch(data_iv_next, arrays["m"], arrays["tau"])
    gen_penalty = arbitrage_penalty_batch(gen_iv.reshape(-1, gen_iv.shape[-1]), arrays["m"], arrays["tau"])
    return {
        "eval_dates": int(len(chosen)),
        "eval_samples_per_date": int(cfg.eval_samples),
        "date_start": str(arrays["dates_t"][chosen[0]].date()),
        "date_end": str(arrays["dates_t"][chosen[-1]].date()),
        "return_data_mean": float(np.mean(data_returns)),
        "return_generated_mean_of_means": float(np.mean(gen_return_mean)),
        "return_mean_abs_error": float(np.mean(np.abs(gen_return_mean - data_returns))),
        "iv_mean_abs_error": float(np.mean(np.abs(gen_iv_mean - data_iv_next))),
        "data_arbitrage_mean": float(np.mean(data_penalty)),
        "data_arbitrage_median": float(np.median(data_penalty)),
        "generated_arbitrage_mean": float(np.mean(gen_penalty)),
        "generated_arbitrage_median": float(np.median(gen_penalty)),
        "generated_arbitrage_p95": float(np.quantile(gen_penalty, 0.95)),
    }


def selection_score(metrics: dict[str, object]) -> float:
    data_arb = max(float(metrics["data_arbitrage_mean"]), 1e-8)
    return (
        float(metrics["generated_arbitrage_mean"]) / data_arb
        + 10.0 * float(metrics["return_mean_abs_error"])
        + float(metrics["iv_mean_abs_error"])
    )


def save_checkpoint(path: Path, gen: Generator, disc: Discriminator, cfg: TrainConfig, arrays: dict[str, object], train_idx: np.ndarray, test_idx: np.ndarray, metrics: dict[str, object], alpha_beta: dict[str, object]) -> None:
    torch.save(
        {
            "gen_state_dict": gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "config": asdict(cfg),
            "m": arrays["m"],
            "tau": arrays["tau"],
            "train_idx": train_idx,
            "test_idx": test_idx,
            "metrics": metrics,
            "gradient_matching": alpha_beta,
        },
        path,
    )


def train_one(processed_dir: Path, out_dir: Path, cfg: TrainConfig, device: torch.device) -> dict[str, object]:
    set_reproducible(cfg.seed)
    arrays = load_arrays(processed_dir)
    train_idx, test_idx = split_indices(arrays["dates_t"], cfg.train_end)
    gen, disc = build_model(arrays, cfg, device)
    condition_train = torch.tensor(arrays["condition"][train_idx], dtype=torch.float32, device=device)
    true_train = torch.tensor(arrays["true"][train_idx], dtype=torch.float32, device=device)
    criterion = nn.BCELoss().to(device)
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=cfg.lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=cfg.lrd)
    alpha, beta, gm = gradient_matching(gen, disc, criterion, gen_opt, disc_opt, condition_train, true_train, arrays["m"], arrays["tau"], cfg, device)
    history = train_main_loop(gen, disc, criterion, gen_opt, disc_opt, condition_train, true_train, arrays["m"], arrays["tau"], alpha, beta, cfg, device)
    metrics = evaluate_model(gen, arrays, test_idx, cfg, device)
    metrics["selection_score"] = selection_score(metrics)
    ckpt = out_dir / f"{cfg.name}.pt"
    save_checkpoint(ckpt, gen, disc, cfg, arrays, train_idx, test_idx, metrics, gm)
    result = {"config": asdict(cfg), "checkpoint": str(ckpt), "gradient_matching": gm, "history": history, "metrics": metrics}
    write_json(out_dir / f"{cfg.name}_summary.json", result)
    return result


def load_checkpoint(path: Path, arrays: dict[str, object], device: torch.device) -> tuple[Generator, TrainConfig, dict[str, object]]:
    ckpt = torch.load(path, map_location=device)
    cfg = TrainConfig(**ckpt["config"])
    gen, _disc = build_model(arrays, cfg, device)
    gen.load_state_dict(ckpt["gen_state_dict"])
    gen.to(device).eval()
    return gen, cfg, ckpt


def current_spot(current_target: pd.DataFrame, current_hedges: pd.DataFrame) -> float:
    for frame in (current_target, current_hedges):
        if "spot" in frame.columns:
            vals = pd.to_numeric(frame["spot"], errors="coerce").dropna()
            if not vals.empty and float(vals.iloc[0]) > 0:
                return float(vals.iloc[0])
    raise ValueError("could not infer current spot from panel quotes")


def bs_price(spot: float, strike: float, tau: float, sigma: float, cp_flag: str, r: float = 0.0) -> float:
    if cp_flag == "U":
        return spot
    tau = max(float(tau), 1.0 / 365.0)
    sigma = max(float(sigma), 1e-6)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    disc = math.exp(-r * tau)
    if cp_flag == "C":
        return spot * cdf(d1) - strike * disc * cdf(d2)
    if cp_flag == "P":
        return strike * disc * cdf(-d2) - spot * cdf(-d1)
    raise ValueError(cp_flag)


def interpolated_iv(
    iv_surface: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    moneyness: float,
    tau: float,
    floor_nonpositive: bool = False,
    diagnostics: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
) -> float:
    """Paper-style linear interpolation/extrapolation on the VolGAN IV grid.

    Generated VolGAN surfaces are positive on-grid, but linear extrapolation can
    become negative in stress-date off-grid repricing.  In that generated-scenario
    path only, floor infeasible extrapolated IVs to the scenario surface's own
    minimum positive grid IV so Black-Scholes repricing remains feasible without
    changing the interpolation rule where it is valid.
    """
    surface = np.asarray(iv_surface, dtype=float).reshape(len(m_grid), len(tau_grid))
    value = interpolate_surface(surface, np.asarray(m_grid, dtype=float), np.asarray(tau_grid, dtype=float), np.array([float(moneyness)]), np.array([float(tau)]))
    sigma = float(np.asarray(value).reshape(-1)[0])
    if np.isfinite(sigma) and sigma > 0:
        return sigma
    if floor_nonpositive:
        positive = surface[np.isfinite(surface) & (surface > 0)]
        if positive.size:
            floor_value = float(np.min(positive))
            if diagnostics is not None:
                diagnostics["generated_iv_floor_count"] = int(diagnostics.get("generated_iv_floor_count", 0)) + 1
                diagnostics["generated_iv_floor_min_raw"] = float(min(float(diagnostics.get("generated_iv_floor_min_raw", sigma)), sigma)) if "generated_iv_floor_min_raw" in diagnostics else float(sigma)
                diagnostics["generated_iv_floor_min_value"] = float(min(float(diagnostics.get("generated_iv_floor_min_value", floor_value)), floor_value)) if "generated_iv_floor_min_value" in diagnostics else float(floor_value)
                diagnostics["generated_iv_floor_max_value"] = float(max(float(diagnostics.get("generated_iv_floor_max_value", floor_value)), floor_value)) if "generated_iv_floor_max_value" in diagnostics else float(floor_value)
                examples = diagnostics.setdefault("generated_iv_floor_examples", [])
                if isinstance(examples, list) and len(examples) < 25:
                    record = {"raw_sigma": float(sigma), "floor_sigma": floor_value, "moneyness": float(moneyness), "tau": float(tau)}
                    if context:
                        record.update(context)
                    examples.append(record)
            return floor_value
    raise ValueError(f"interpolated IV is not positive finite: {sigma}")


def _bs_price_vec(spot, strike, tau, sigma, cp_flag, r=0.0):
    """Vectorized Black-Scholes price.

    All inputs are numpy arrays of the same broadcastable shape except cp_flag,
    which is an array of bytes/str/object with values in {'C','P','U'}.  For
    'U' rows the spot is returned.  Numerical conventions match the scalar
    :func:`bs_price` exactly: tau is floored to 1/365, sigma is floored to
    1e-6, and N(.) is computed via ``scipy.special.ndtr`` which equals
    ``0.5*(1+erf(x/sqrt(2)))``.
    """
    from scipy.special import ndtr
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if isinstance(cp_flag, np.ndarray):
        cp = cp_flag
    else:
        cp = np.asarray(cp_flag)
    cp = np.array([str(x).upper() for x in cp.reshape(-1)], dtype="<U1").reshape(cp.shape)
    if not isinstance(r, np.ndarray):
        r = np.full_like(spot, float(r))
    tau_f = np.maximum(tau, 1.0 / 365.0)
    sig_f = np.maximum(sigma, 1e-6)
    sqrt_tau = np.sqrt(tau_f)
    # NaN-safe: where sigma or strike is non-positive/NaN, propagate NaN.
    bad = ~(np.isfinite(sigma) & (sigma > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (r + 0.5 * sig_f * sig_f) * tau_f) / (sig_f * sqrt_tau)
        d2 = d1 - sig_f * sqrt_tau
        disc = np.exp(-r * tau_f)
        call = spot * ndtr(d1) - strike * disc * ndtr(d2)
        put = strike * disc * ndtr(-d2) - spot * ndtr(-d1)
    is_call = cp == "C"
    is_put = cp == "P"
    is_u = cp == "U"
    out = np.where(is_call, call, np.where(is_put, put, np.where(is_u, spot, np.nan)))
    out = np.where(bad & ~is_u, np.nan, out)
    return out


def _interpolated_iv_vec(surfaces, m_grid, tau_grid, moneyness, tau):
    """Vectorized bilinear interpolation/extrapolation on the IV grid.

    Parameters
    ----------
    surfaces : array, shape (K, len(m_grid), len(tau_grid))
        IV surfaces.
    m_grid, tau_grid : 1-D sorted arrays.
    moneyness, tau : arrays, shape (K, N) or broadcastable.
        Per-scenario, per-contract query points.

    Returns
    -------
    iv : array, shape (K, N)
        Interpolated IV.  Mathematically equivalent to the scalar
        :func:`interpolated_iv` (without the floor logic — apply the floor
        separately).
    """
    surfaces = np.asarray(surfaces, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    moneyness = np.asarray(moneyness, dtype=float)
    tau = np.asarray(tau, dtype=float)
    nm = m_grid.shape[0]
    nt = tau_grid.shape[0]
    # Use searchsorted, then clip to [1, n-1] so endpoints/extrapolation reuse
    # the boundary segment slope (matches scipy.interp1d fill_value='extrapolate').
    i_m = np.clip(np.searchsorted(m_grid, moneyness, side="right"), 1, nm - 1)
    i_t = np.clip(np.searchsorted(tau_grid, tau, side="right"), 1, nt - 1)
    m0 = m_grid[i_m - 1]
    m1 = m_grid[i_m]
    t0 = tau_grid[i_t - 1]
    t1 = tau_grid[i_t]
    w_m = (moneyness - m0) / (m1 - m0)
    w_t = (tau - t0) / (t1 - t0)
    # Gather the four corners.  surfaces[k, i_m-1, i_t-1] etc., with k indexed
    # along the leading axis of moneyness/tau.
    K = surfaces.shape[0]
    k_idx = np.arange(K).reshape((K,) + (1,) * (moneyness.ndim - 1))
    v00 = surfaces[k_idx, i_m - 1, i_t - 1]
    v10 = surfaces[k_idx, i_m,     i_t - 1]
    v01 = surfaces[k_idx, i_m - 1, i_t]
    v11 = surfaces[k_idx, i_m,     i_t]
    iv = (1 - w_m) * (1 - w_t) * v00 + w_m * (1 - w_t) * v10 + (1 - w_m) * w_t * v01 + w_m * w_t * v11
    return iv


def _apply_iv_floor(iv, surfaces, diagnostics=None, contexts=None):
    """Apply the ``floor_nonpositive`` logic used by interpolated_iv.

    For every (k, n) entry where ``iv`` is not finite-and-positive, replace it
    with the minimum positive finite value of ``surfaces[k]``.  Mirrors the
    diagnostics bookkeeping of the scalar path.
    """
    iv = np.asarray(iv, dtype=float).copy()
    K = surfaces.shape[0]
    floors = np.full(K, np.nan, dtype=float)
    for k in range(K):
        s = surfaces[k]
        positive = s[np.isfinite(s) & (s > 0)]
        if positive.size:
            floors[k] = float(np.min(positive))
    bad = ~(np.isfinite(iv) & (iv > 0))
    if bad.any():
        # broadcast floors along the trailing axis
        floor_b = np.broadcast_to(floors.reshape((K,) + (1,) * (iv.ndim - 1)), iv.shape)
        # Indices to update only where the floor is finite (else raise like the scalar path)
        floor_ok = bad & np.isfinite(floor_b)
        floor_bad = bad & ~np.isfinite(floor_b)
        if floor_bad.any():
            raise ValueError("interpolated IV is not positive finite and floor unavailable")
        if diagnostics is not None and floor_ok.any():
            # Bookkeeping mirrors the scalar path's running statistics.
            count = int(floor_ok.sum())
            diagnostics["generated_iv_floor_count"] = int(diagnostics.get("generated_iv_floor_count", 0)) + count
            raw_vals = iv[floor_ok]
            flv = floor_b[floor_ok]
            if raw_vals.size:
                cur_min_raw = float(diagnostics.get("generated_iv_floor_min_raw", np.min(raw_vals)))
                diagnostics["generated_iv_floor_min_raw"] = float(min(cur_min_raw, float(np.min(raw_vals))))
                cur_min_val = float(diagnostics.get("generated_iv_floor_min_value", np.min(flv)))
                diagnostics["generated_iv_floor_min_value"] = float(min(cur_min_val, float(np.min(flv))))
                cur_max_val = float(diagnostics.get("generated_iv_floor_max_value", np.max(flv)))
                diagnostics["generated_iv_floor_max_value"] = float(max(cur_max_val, float(np.max(flv))))
            if contexts is not None:
                examples = diagnostics.setdefault("generated_iv_floor_examples", [])
                if isinstance(examples, list):
                    idxs = np.argwhere(floor_ok)
                    for row in idxs:
                        if len(examples) >= 25:
                            break
                        ctx = contexts(*row)
                        rec = {"raw_sigma": float(iv[tuple(row)]), "floor_sigma": float(floor_b[tuple(row)]),
                               "moneyness": float(ctx.get("moneyness", np.nan)),
                               "tau": float(ctx.get("tau", np.nan))}
                        rec.update({k: v for k, v in ctx.items() if k not in {"moneyness", "tau"}})
                        examples.append(rec)
        iv[floor_ok] = floor_b[floor_ok]
    return iv


def contract_tau_at(contract: pd.Series, date: pd.Timestamp) -> float:
    if "exdate" in contract.index and pd.notna(contract["exdate"]):
        days = (pd.Timestamp(contract["exdate"]).normalize() - pd.Timestamp(date).normalize()).days
        return max(days / 365.0, 1.0 / 365.0)
    if "days_to_exp" in contract.index and pd.notna(contract["days_to_exp"]):
        return max(float(contract["days_to_exp"]) / 365.0 - 1.0 / 365.0, 1.0 / 365.0)
    if "tau" in contract.index and pd.notna(contract["tau"]):
        return max(float(contract["tau"]) - 1.0 / 365.0, 1.0 / 365.0)
    raise ValueError("contract needs exdate, days_to_exp, or tau")


def add_underlying_to_panel(panel: HedgePanel, data_dir: Path) -> HedgePanel:
    """Phase 4 / M10: idempotent shim.

    ``hedging.build_instrument_panel`` now injects the UNDERLYING_SPX row and
    its per-date quotes directly, so this helper is kept only for backwards
    compatibility with ``recompute_benchmark_fixed.py`` and similar callers.
    If the SPX row is already present the panel is returned unchanged.
    """
    if "UNDERLYING_SPX" in set(panel.hedges["optionid"].astype(str)):
        return panel
    raise RuntimeError(
        "add_underlying_to_panel called on a panel without UNDERLYING_SPX; "
        "hedging.build_instrument_panel should have injected it (Phase 4 / M10)."
    )


def trim_panel_liquid_hedges(panel: HedgePanel, min_observed_frac: float) -> HedgePanel:
    """Keep SPX plus option hedges with enough observed quote coverage.

    The target straddle is not trimmed; missing target quote intervals remain
    skipped by run_daily_backtest.  This is an experiment-layer accommodation
    for the positive-volume raw option filter.
    """
    from hedging import quote_coverage

    if min_observed_frac <= 0:
        return panel
    coverage = panel.missing_quotes.copy()
    hedge_cov = coverage[coverage["role"] == "hedge"].copy()
    if hedge_cov.empty:
        return panel
    hedge_cov["observed_frac"] = hedge_cov["observed_dates"] / hedge_cov["expected_dates"].clip(lower=1)
    keep_ids = set(hedge_cov.loc[hedge_cov["observed_frac"] >= min_observed_frac, "optionid"])
    keep_ids.add("UNDERLYING_SPX")
    hedges = panel.hedges[panel.hedges["optionid"].isin(keep_ids)].copy().reset_index(drop=True)
    quotes = panel.quotes[(panel.quotes["role"] != "hedge") | (panel.quotes["optionid"].isin(keep_ids))].copy().reset_index(drop=True)
    missing = quote_coverage(quotes, pd.concat([panel.target, hedges], ignore_index=True, sort=False), panel.trading_dates)
    return HedgePanel(panel.start_date, panel.expiry_date, panel.m0, panel.target, hedges, quotes, missing, panel.trading_dates)


class VolGANScenarioSource:
    def __init__(self, gen: Generator, cfg: TrainConfig, arrays: dict[str, object], device: torch.device, samples: int, validation_samples: int = 0):
        self.gen = gen
        self.cfg = cfg
        self.arrays = arrays
        self.device = device
        self.samples = samples
        self.validation_samples = validation_samples
        self.generated_iv_diagnostics: dict[str, object] = {"generated_iv_floor_count": 0}
        self.date_to_index = {pd.Timestamp(d).normalize(): i for i, d in enumerate(arrays["dates_t"])}

    def _direct_scenarios_for_k(self, end_date, current_target, current_hedges, k: int) -> DirectScenarioChanges:
        end_date = pd.Timestamp(end_date).normalize()
        if end_date not in self.date_to_index:
            raise ValueError(f"VolGAN condition not available for end date {end_date.date()}")
        idx = self.date_to_index[end_date]
        cond = self.arrays["condition"][idx]
        gen_returns, gen_iv = generate_for_conditions(self.gen, cond.reshape(1, -1), self.cfg, k, self.device)
        returns = gen_returns[0]
        m_grid = np.asarray(self.arrays["m"], dtype=float)
        tau_grid = np.asarray(self.arrays["tau"], dtype=float)
        ivs = gen_iv[0].reshape(k, len(m_grid), len(tau_grid))
        s0 = current_spot(current_target, current_hedges)
        spots_next = s0 * np.exp(returns)  # (k,)

        current_target_values = pd.to_numeric(current_target["mid_price"], errors="coerce").to_numpy(dtype=float)
        current_hedge_values = pd.to_numeric(current_hedges["mid_price"], errors="coerce").to_numpy(dtype=float)

        # --- Materialize per-contract arrays once ---
        def _contract_arrays(df):
            strikes = pd.to_numeric(df["strike"], errors="coerce").to_numpy(dtype=float)
            taus = np.array([contract_tau_at(row, end_date) for _, row in df.reset_index(drop=True).iterrows()], dtype=float)
            cp = np.array([str(row.get("cp_flag", "")).upper() if hasattr(row, "get") else str(row["cp_flag"]).upper()
                           for _, row in df.reset_index(drop=True).iterrows()], dtype="<U1")
            return strikes, taus, cp

        tgt_strikes, tgt_taus, tgt_cp = _contract_arrays(current_target)
        hed_strikes, hed_taus, hed_cp = _contract_arrays(current_hedges)

        # --- Vectorized repricing for target ---
        def _reprice(strikes, taus, cp_arr, role):
            n = strikes.shape[0]
            if n == 0:
                return np.zeros((k, 0), dtype=float)
            spot_kn = spots_next.reshape(k, 1)  # (k,1)
            strike_kn = np.broadcast_to(strikes.reshape(1, n), (k, n))
            tau_kn = np.broadcast_to(taus.reshape(1, n), (k, n))
            cp_kn = np.broadcast_to(cp_arr.reshape(1, n), (k, n))
            moneyness_kn = strike_kn / spot_kn
            # Underlying rows (cp == 'U') get spot directly; for them, IV value
            # is irrelevant, but we still must not have it propagate NaNs into
            # the non-U path.  We mask before/after the interp.
            non_u_mask = cp_kn != "U"
            iv_kn = _interpolated_iv_vec(ivs, m_grid, tau_grid, moneyness_kn, tau_kn)
            # Build a context callable for diagnostics floor examples.
            def _ctx(ki, ni):
                return {"end_date": str(end_date.date()), "sample_idx": int(ki), "role": role, "contract_idx": int(ni),
                        "moneyness": float(moneyness_kn[ki, ni]), "tau": float(tau_kn[ki, ni])}
            # Only floor on the non-underlying rows; underlying rows can keep NaN IV (unused).
            iv_for_floor = np.where(non_u_mask, iv_kn, np.nan)
            iv_floored = _apply_iv_floor(iv_for_floor, ivs, diagnostics=self.generated_iv_diagnostics, contexts=_ctx) if non_u_mask.any() else iv_kn
            prices = _bs_price_vec(spot_kn, strike_kn, tau_kn, iv_floored, cp_kn)
            # _bs_price_vec already handles 'U' → spot.
            return prices

        next_target = _reprice(tgt_strikes, tgt_taus, tgt_cp, "target")
        next_hedges = _reprice(hed_strikes, hed_taus, hed_cp, "hedge")

        return DirectScenarioChanges(
            target_changes=next_target.sum(axis=1) - float(np.sum(current_target_values)),
            hedge_changes=next_hedges - current_hedge_values.reshape(1, -1),
        )

    def scenarios_for_interval(self, panel, start_date, end_date, current_target, current_hedges):
        train = self._direct_scenarios_for_k(end_date, current_target, current_hedges, self.samples)
        if self.validation_samples > 0:
            validation = self._direct_scenarios_for_k(end_date, current_target, current_hedges, self.validation_samples)
            return {"train": train, "validation": validation}
        return train

def non_overlapping_start_dates(data_dir: Path, start: str, end: str, max_periods: int | None, schedule: str = "calendar31") -> list[pd.Timestamp]:
    from hedging import first_trading_date_on_or_after, load_underlying

    underlying = load_underlying(data_dir=data_dir, start_date=start, end_date=end)
    if underlying.empty:
        return []
    trading_dates = pd.to_datetime(underlying["date"]).sort_values().reset_index(drop=True)
    if schedule == "paper_23td":
        starts = list(trading_dates.iloc[3::23])
        if max_periods:
            starts = starts[:max_periods]
        return [pd.Timestamp(x).normalize() for x in starts]
    if schedule != "calendar31":
        raise ValueError(f"unknown hedge schedule {schedule}")
    starts = []
    cursor = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    while cursor <= end_ts:
        actual = first_trading_date_on_or_after(cursor, underlying)
        if actual > end_ts:
            break
        starts.append(actual)
        if max_periods and len(starts) >= max_periods:
            break
        cursor = actual + pd.Timedelta(days=31)
    return starts



def parse_position_vector(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        return np.asarray([], dtype=float)
    cleaned = str(value).replace("[", " ").replace("]", " ").replace(",", " ")
    return np.fromstring(cleaned, sep=" ")


def required_paper_columns() -> set[str]:
    return {"m0", "period_start", "period_expiry", "start_date", "end_date", "strategy", "positions", "target_change", "realized_tracking_error", "cumulative_z_period"}


def normalize_paper_results(results: pd.DataFrame) -> pd.DataFrame:
    missing = required_paper_columns() - set(results.columns)
    if missing:
        raise ValueError(f"paper report missing required columns: {sorted(missing)}")
    df = results.copy()
    for col in ("period_start", "period_expiry", "start_date", "end_date"):
        df[col] = pd.to_datetime(df[col])
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    # Paper Eq (6): Z_t = V_t - Pi_t is CUMULATIVE within each one-month period (Z_0 = 0).
    # Paper Table 2 / Figs 6, 11, 12, 13 stats use this cumulative quantity, not the daily increment.
    df["z_t"] = pd.to_numeric(df["cumulative_z_period"], errors="coerce")
    df["z_t_daily"] = pd.to_numeric(df["realized_tracking_error"], errors="coerce")
    base_cols = ["m0", "period_start", "period_expiry", "start_date", "end_date", "target_change"]
    unhedged_rows = df[base_cols].drop_duplicates(subset=["m0", "period_start", "period_expiry", "start_date", "end_date"]).copy()
    # Unhedged Z_t = V_t (cumulative target change within (m0, period_start), starting at 0 on period_start).
    unhedged_rows = unhedged_rows.sort_values(["m0", "period_start", "end_date"]).copy()
    unhedged_rows["z_t_daily"] = pd.to_numeric(unhedged_rows["target_change"], errors="coerce")
    unhedged_rows["z_t"] = unhedged_rows.groupby(["m0", "period_start"])["z_t_daily"].cumsum()
    unhedged_rows["strategy"] = "unhedged"
    unhedged_rows["strategy_label"] = STRATEGY_LABELS["unhedged"]
    unhedged_rows["positions"] = "[]"
    keep = ["m0", "period_start", "period_expiry", "start_date", "end_date", "strategy", "strategy_label", "positions", "z_t", "z_t_daily"]
    return pd.concat([df[keep], unhedged_rows[keep]], ignore_index=True, sort=False).dropna(subset=["z_t"])


def add_sample_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    covid_mask = (out["period_start"] >= PAPER_COVID_START) & (out["period_start"] <= PAPER_COVID_END)
    out["covid_excluded"] = ~covid_mask
    return out


def stats_for_values(values: pd.Series) -> dict[str, float | int]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "variance": np.nan, "std": np.nan, "var_5": np.nan, "var_2_5": np.nan, "var_1": np.nan}
    ddof = 1 if arr.size > 1 else 0
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "variance": float(np.var(arr, ddof=ddof)),
        "std": float(np.std(arr, ddof=ddof)),
        "var_5": float(-np.quantile(arr, 0.05)),
        "var_2_5": float(-np.quantile(arr, 0.025)),
        "var_1": float(-np.quantile(arr, 0.01)),
    }


def paper_stats_table(df: pd.DataFrame, by_m0: bool) -> pd.DataFrame:
    rows = []
    for sample, sample_df in (("full", df), ("covid_excluded", df[df["covid_excluded"]])):
        group_cols = ["strategy", "strategy_label"] + (["m0"] if by_m0 else [])
        for keys, group in sample_df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"sample": sample}
            for col, value in zip(group_cols, keys):
                row[col] = value
            row.update(stats_for_values(group["z_t"]))
            rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    strategy_order = {"Unhedged": 0, "Delta": 1, "Delta-vega": 2, "Data-driven": 3}
    table["_strategy_order"] = table["strategy_label"].map(strategy_order).fillna(99)
    sort_cols = ["sample"] + (["m0"] if by_m0 else []) + ["_strategy_order"]
    return table.sort_values(sort_cols).drop(columns=["_strategy_order"]).reset_index(drop=True)


def paper_table2_comparison(pooled: pd.DataFrame) -> pd.DataFrame:
    targets = pd.DataFrame(PAPER_TABLE2_TARGETS)
    observed = pooled.drop(columns=["strategy"], errors="ignore").copy()
    merged = targets.merge(observed, on=["sample", "strategy_label"], suffixes=("_paper", "_observed"), how="left")
    metrics = ["mean", "median", "std", "var_5", "var_2_5", "var_1"]
    for metric in metrics:
        merged[f"{metric}_abs_diff"] = merged[f"{metric}_observed"] - merged[f"{metric}_paper"]
        denom = merged[f"{metric}_paper"].replace(0, np.nan).abs()
        merged[f"{metric}_rel_diff"] = merged[f"{metric}_abs_diff"] / denom
    return merged


def prepare_paper_figure_inputs(df: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    inputs_dir = figure_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    lasso = df[df["strategy"] == "lasso"].copy()
    if not lasso.empty:
        lasso["selected_count"] = lasso["positions"].map(lambda x: int(np.count_nonzero(np.abs(parse_position_vector(x)) > 1e-8)))
        fig05 = lasso.groupby(["period_start", "m0"], as_index=False)["selected_count"].mean()
        path = inputs_dir / "fig05_selected_counts.csv"
        fig05.to_csv(path, index=False)
        paths["fig05"] = str(path)
        fig11 = lasso[["end_date", "m0", "z_t"]].copy()
        path = inputs_dir / "fig11_12_lasso_timeseries.csv"
        fig11.to_csv(path, index=False)
        paths["fig11_12"] = str(path)
        fig13 = lasso[["m0", "z_t"]].copy()
        path = inputs_dir / "fig13_lasso_distribution_by_m0.csv"
        fig13.to_csv(path, index=False)
        paths["fig13"] = str(path)
    path = inputs_dir / "fig06_distribution_all_strategies.csv"
    df[["strategy", "strategy_label", "m0", "period_start", "end_date", "z_t"]].to_csv(path, index=False)
    paths["fig06"] = str(path)
    keyed = df[df["strategy"].isin(["lasso", "delta", "delta_vega"])].pivot_table(
        index=["m0", "period_start", "start_date", "end_date"], columns="strategy", values="z_t", aggfunc="first"
    ).reset_index()
    keyed["covid_excluded"] = ~((keyed["period_start"] >= PAPER_COVID_START) & (keyed["period_start"] <= PAPER_COVID_END))
    path = inputs_dir / "fig07_10_scatter_inputs.csv"
    keyed.to_csv(path, index=False)
    paths["fig07_10"] = str(path)
    return paths


def render_paper_figures(df: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    lasso = df[df["strategy"] == "lasso"].copy()
    if not lasso.empty:
        lasso["selected_count"] = lasso["positions"].map(lambda x: int(np.count_nonzero(np.abs(parse_position_vector(x)) > 1e-8)))
        counts = lasso.groupby(["period_start", "m0"], as_index=False)["selected_count"].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        for m0, group in counts.groupby("m0"):
            ax.plot(group["period_start"], group["selected_count"], marker="o", linewidth=1.2, label=f"m0={m0:g}")
        ax.set_title("Fig. 5: selected VolGAN hedge instruments over time")
        ax.set_xlabel("Position start date")
        ax.set_ylabel("Selected instruments")
        ax.legend(ncol=3, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = figure_dir / "fig05_selected_instruments.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["fig05"] = str(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in df.groupby("strategy_label"):
        if len(group) > 1:
            ax.hist(group["z_t"], bins=60, density=True, alpha=0.35, label=label)
    ax.set_title("Fig. 6: distribution of tracking error Z_t")
    ax.set_xlabel("Z_t")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "fig06_zt_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["fig06"] = str(path)

    keyed = df[df["strategy"].isin(["lasso", "delta", "delta_vega"])].pivot_table(
        index=["m0", "period_start", "start_date", "end_date"], columns="strategy", values="z_t", aggfunc="first"
    ).reset_index()
    keyed["covid_excluded"] = ~((keyed["period_start"] >= PAPER_COVID_START) & (keyed["period_start"] <= PAPER_COVID_END))

    def scatter_figure(xcol: str, subset: pd.DataFrame, title: str, filename: str) -> None:
        if {xcol, "lasso"} - set(subset.columns):
            return
        plot_df = subset.dropna(subset=[xcol, "lasso"])
        if plot_df.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(plot_df[xcol], plot_df["lasso"], s=16, alpha=0.65)
        lo = float(np.nanmin([plot_df[xcol].min(), plot_df["lasso"].min()]))
        hi = float(np.nanmax([plot_df[xcol].max(), plot_df["lasso"].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(STRATEGY_LABELS.get(xcol, xcol))
        ax.set_ylabel("Data-driven")
        fig.tight_layout()
        out = figure_dir / filename
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths[filename[:5]] = str(out)

    scatter_figure("delta", keyed[keyed["covid_excluded"]], "Fig. 7: Delta vs data-driven, Covid excluded", "fig07_delta_scatter_no_covid.png")
    scatter_figure("delta_vega", keyed[keyed["covid_excluded"]], "Fig. 8: Delta-vega vs data-driven, Covid excluded", "fig08_delta_vega_scatter_no_covid.png")
    scatter_figure("delta", keyed, "Fig. 9: Delta vs data-driven, Covid included", "fig09_delta_scatter_full.png")
    scatter_figure("delta_vega", keyed, "Fig. 10: Delta-vega vs data-driven, Covid included", "fig10_delta_vega_scatter_full.png")

    if not lasso.empty:
        m0_values = sorted(lasso["m0"].dropna().unique())
        fig, axes = plt.subplots(len(m0_values), 1, figsize=(10, max(2.2 * len(m0_values), 4)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, m0 in zip(axes, m0_values):
            group = lasso[lasso["m0"] == m0].sort_values("end_date")
            ax.plot(group["end_date"], group["z_t"], linewidth=1.0)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel(f"m0={m0:g}")
        axes[-1].set_xlabel("Date")
        fig.suptitle("Fig. 11: data-driven tracking error time series by m0")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = figure_dir / "fig11_lasso_timeseries_by_m0.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["fig11"] = str(path)

        fig, axes = plt.subplots(len(m0_values), 1, figsize=(10, max(2.2 * len(m0_values), 4)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, m0 in zip(axes, m0_values):
            group = lasso[lasso["m0"] == m0].sort_values("end_date")
            ax.plot(group["end_date"], group["z_t"], linewidth=1.0)
            ax.set_yscale("symlog", linthresh=50)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axhspan(-50, 50, color="grey", alpha=0.08)
            ax.set_ylabel(f"m0={m0:g}")
        axes[-1].set_xlabel("Date")
        fig.suptitle("Fig. 12: data-driven tracking error, symlog outside [-50, 50]")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = figure_dir / "fig12_lasso_timeseries_symlog.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["fig12"] = str(path)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, m0 in zip(axes, m0_values):
            group = lasso[lasso["m0"] == m0]
            ax.hist(group["z_t"], bins=35, density=True, alpha=0.7)
            ax.set_title(f"m0={m0:g}")
            ax.axvline(0.0, color="black", linewidth=0.8)
        for ax in axes[len(m0_values):]:
            ax.axis("off")
        fig.suptitle("Fig. 13: data-driven Z_t distributions by m0")
        fig.supxlabel("Z_t")
        fig.supylabel("Density")
        fig.tight_layout()
        path = figure_dir / "fig13_lasso_distributions_by_m0.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["fig13"] = str(path)
    return paths


def write_paper_outputs(results: pd.DataFrame, out_dir: Path, metadata: dict[str, object]) -> dict[str, object]:
    paper_dir = out_dir / "paper_reproduction"
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = add_sample_flags(normalize_paper_results(results))
    df.to_csv(paper_dir / "paper_tracking_errors_long.csv", index=False)
    pooled = paper_stats_table(df, by_m0=False)
    by_m0 = paper_stats_table(df, by_m0=True)
    comparison = paper_table2_comparison(pooled)
    pooled.to_csv(tables_dir / "table2_pooled_stats_observed.csv", index=False)
    by_m0.to_csv(tables_dir / "tables5_7_by_m0_stats_observed.csv", index=False)
    comparison.to_csv(tables_dir / "table2_vs_paper_targets.csv", index=False)
    figure_inputs = prepare_paper_figure_inputs(df, figures_dir)
    figures = render_paper_figures(df, figures_dir)
    unique_periods = int(df.loc[df["strategy"] == "lasso", "period_start"].nunique())
    min_periods_by_m0 = int(df[df["strategy"] == "lasso"].groupby("m0")["period_start"].nunique().min()) if not df[df["strategy"] == "lasso"].empty else 0
    meta = {
        **metadata,
        "paper_sources": [
            "https://link.springer.com/article/10.1007/s10479-025-06867-3",
            "https://link.springer.com/article/10.1007/s10479-025-06867-3/tables/2",
        ],
        "paper_expected_periods": PAPER_EXPECTED_PERIODS,
        "observed_lasso_unique_periods": unique_periods,
        "observed_lasso_min_periods_by_m0": min_periods_by_m0,
        "date_min": str(df["end_date"].min().date()) if not df.empty else None,
        "date_max": str(df["end_date"].max().date()) if not df.empty else None,
        "covid_exclusion_period_start_window": [str(PAPER_COVID_START.date()), str(PAPER_COVID_END.date())],
        "var_convention": "positive lower-tail loss magnitude: -quantile(Z_t, q)",
        "std_convention": "sample standard deviation, ddof=1 when count > 1",
        "completeness_status": "FULL_PERIOD_CANDIDATE" if min_periods_by_m0 >= PAPER_EXPECTED_PERIODS else "INCOMPLETE_CAPPED_OR_SPARSE",
        "figure_inputs": figure_inputs,
        "figures": figures,
        "tables": {
            "pooled": str(tables_dir / "table2_pooled_stats_observed.csv"),
            "by_m0": str(tables_dir / "tables5_7_by_m0_stats_observed.csv"),
            "table2_comparison": str(tables_dir / "table2_vs_paper_targets.csv"),
        },
    }
    write_json(paper_dir / "paper_reproduction_metadata.json", meta)
    return meta


def run_hedging(checkpoint: Path, processed_dir: Path, data_dir: Path, out_dir: Path, device: torch.device, samples: int, validation_samples: int, m0_values: list[float], start: str, end: str, max_periods: int | None, min_hedge_observed_frac: float, hedge_schedule: str = "calendar31", hedge_quote_source: str = "observed") -> dict[str, object]:
    if samples <= 0:
        raise ValueError("--hedge-samples must be positive")
    if validation_samples < 0:
        raise ValueError("--hedge-validation-samples must be nonnegative")
    arrays = load_arrays(processed_dir)
    gen, cfg, ckpt = load_checkpoint(checkpoint, arrays, device)
    scenario_source = VolGANScenarioSource(gen, cfg, arrays, device, samples, validation_samples)
    if hedge_quote_source not in {"observed", "surface"}:
        raise ValueError(f"unsupported hedge_quote_source: {hedge_quote_source}")
    surface_market = load_smoothed_surface_market(processed_dir) if hedge_quote_source == "surface" else None
    starts = non_overlapping_start_dates(data_dir, start, end, max_periods, hedge_schedule)
    summaries = []
    result_frames = []
    skipped_frames = []
    for m0 in m0_values:
        for start_date in starts:
            try:
                panel = build_instrument_panel(start_date=start_date, m0=m0, data_dir=data_dir)
                original_hedge_count = len(panel.hedges)
                # Phase 4 / M10: build_instrument_panel now injects UNDERLYING_SPX directly.
                if hedge_quote_source == "observed":
                    panel = trim_panel_liquid_hedges(panel, min_hedge_observed_frac)
                retained_hedge_count = len(panel.hedges)
                summary = run_daily_backtest(panel, scenario_source, alpha_grid=DEFAULT_ALPHA_GRID, quote_source=surface_market)
            except Exception as exc:
                summaries.append({"m0": m0, "start_date": str(pd.Timestamp(start_date).date()), "status": "FAIL", "error": repr(exc)})
                continue
            def _idempotent_insert(df, pos, col, value):
                if col in df.columns:
                    return df
                df.insert(pos, col, value)
                return df
            if not summary.results.empty:
                frame = summary.results.copy()
                frame = _idempotent_insert(frame, 0, "m0", m0)
                frame = _idempotent_insert(frame, 1, "period_start", panel.start_date)
                frame = _idempotent_insert(frame, 2, "period_expiry", panel.expiry_date)
                result_frames.append(frame)
            if not summary.skipped_intervals.empty:
                skipped = summary.skipped_intervals.copy()
                skipped = _idempotent_insert(skipped, 0, "m0", m0)
                skipped = _idempotent_insert(skipped, 1, "period_start", panel.start_date)
                skipped = _idempotent_insert(skipped, 2, "period_expiry", panel.expiry_date)
                skipped_frames.append(skipped)
            summaries.append({
                "m0": m0,
                "start_date": str(panel.start_date.date()),
                "expiry_date": str(panel.expiry_date.date()),
                "status": "PASS",
                "rows": int(summary.results.shape[0]),
                "skipped_intervals": int(summary.skipped_interval_count),
                "valuation_dates": int(len(panel.trading_dates)),
                "pnl_intervals": int(max(0, len(panel.trading_dates) - 1)),
                "unwind_mode": "not_separately_modeled; adjacent-date PnL intervals through selected expiry",
                "original_option_hedges": int(original_hedge_count),
                "retained_hedges_including_spx": int(retained_hedge_count),
            })
            print(f"hedge m0={m0} start={panel.start_date.date()} rows={summary.results.shape[0]} skipped={summary.skipped_interval_count}", flush=True)
    all_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    all_skipped = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    all_results.to_pickle(out_dir / "hedging_results.pkl")
    all_results.to_csv(out_dir / "hedging_results.csv", index=False)
    all_skipped.to_csv(out_dir / "hedging_skipped.csv", index=False)
    combined = BacktestSummary(all_results, all_skipped, int(all_skipped[["start_date", "end_date"]].drop_duplicates().shape[0]) if {"start_date", "end_date"}.issubset(all_skipped.columns) else 0)
    tables = {
        "tracking_error_summary": tracking_error_summary(combined).to_dict(orient="records"),
        "transaction_cost_summary": transaction_cost_summary(combined).to_dict(orient="records"),
        "activity_summary": selected_hedge_count_turnover_summary(combined).to_dict(orient="records"),
        "greek_residual_summary": greek_residual_summary(combined).to_dict(orient="records"),
        "strategy_comparison_table": strategy_comparison_table(combined).to_dict(orient="records"),
        "skip_summary": skip_summary(combined).to_dict(orient="records"),
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    paper_outputs = write_paper_outputs(
        all_results,
        out_dir,
        {
            "checkpoint": str(checkpoint),
            "samples_per_rebalance": samples,
            "validation_samples_per_rebalance": validation_samples,
            "explicit_train_validation_scenarios": validation_samples > 0,
            "min_hedge_observed_frac": min_hedge_observed_frac if hedge_quote_source == "observed" else None,
            "hedge_quote_source": hedge_quote_source,
            "surface_valuation_method": "smoothed IV and half-spread surfaces with datacleaning.interpolate_surface" if hedge_quote_source == "surface" else None,
            "exact_observed_selected_contract_quotes_required": hedge_quote_source == "observed",
            "surface_lookup_diagnostics": surface_market.diagnostics() if surface_market is not None else None,
            "generated_iv_repricing_diagnostics": scenario_source.generated_iv_diagnostics,
            "hedge_schedule": hedge_schedule,
            "m0_values": m0_values,
            "scheduled_period_starts": [str(x.date()) for x in starts],
            "scheduled_period_count_per_m0": len(starts),
            "checkpoint_metrics": ckpt.get("metrics", {}),
        },
    ) if not all_results.empty else {}
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "samples_per_rebalance": samples,
        "validation_samples_per_rebalance": validation_samples,
        "explicit_train_validation_scenarios": validation_samples > 0,
        "min_hedge_observed_frac": min_hedge_observed_frac if hedge_quote_source == "observed" else None,
        "hedge_quote_source": hedge_quote_source,
        "surface_valuation_method": "smoothed IV and half-spread surfaces with datacleaning.interpolate_surface" if hedge_quote_source == "surface" else None,
        "exact_observed_selected_contract_quotes_required": hedge_quote_source == "observed",
        "surface_lookup_diagnostics": surface_market.diagnostics() if surface_market is not None else None,
        "generated_iv_repricing_diagnostics": scenario_source.generated_iv_diagnostics,
        "hedge_schedule": hedge_schedule,
        "m0_values": m0_values,
        "period_starts": [str(x.date()) for x in starts],
        "period_count_per_m0": len(starts),
        "panel_runs": summaries,
        "tables": tables,
        "paper_outputs": paper_outputs,
        "overclaim_boundary": "Shared 11x9 grid and local OptionMetrics preprocessing; not an exact 10x8 VolGAN paper table replication.",
    }
    write_json(out_dir / "hedging_report.json", report)
    return report


def self_check(processed_dir: Path, data_dir: Path, device: torch.device, out_dir: Path) -> None:
    summary = SharedGridVolGANSmokeCheck(processed_dir)
    arrays = load_arrays(processed_dir)
    cfg = CONFIGS["smoke"]
    gen, disc = build_model(arrays, cfg, device)
    train_idx, _ = split_indices(arrays["dates_t"], cfg.train_end)
    condition_train = torch.tensor(arrays["condition"][train_idx[:128]], dtype=torch.float32, device=device)
    true_train = torch.tensor(arrays["true"][train_idx[:128]], dtype=torch.float32, device=device)
    criterion = nn.BCELoss().to(device)
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=cfg.lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=cfg.lrd)
    alpha, beta, gm = gradient_matching(gen, disc, criterion, gen_opt, disc_opt, condition_train, true_train, arrays["m"], arrays["tau"], cfg, device)
    train_main_loop(gen, disc, criterion, gen_opt, disc_opt, condition_train, true_train, arrays["m"], arrays["tau"], alpha, beta, cfg, device)
    metrics = evaluate_model(gen, arrays, np.arange(128, 136), cfg, device)
    panel = build_instrument_panel("2020-01-17", 1.0, data_dir=data_dir)
    # Phase 4 / M10: build_instrument_panel now injects UNDERLYING_SPX directly.
    src = VolGANScenarioSource(gen, cfg, arrays, device, samples=8, validation_samples=2)
    first_date = panel.trading_dates[0]
    second_date = panel.trading_dates[1]
    target_ids = list(panel.target["optionid"])
    hedge_ids = list(panel.hedges["optionid"])
    current_target = panel.quotes[(panel.quotes["date"] == first_date) & panel.quotes["optionid"].isin(target_ids)].drop_duplicates("optionid").set_index("optionid").reindex(target_ids).reset_index()
    current_hedges = panel.quotes[(panel.quotes["date"] == first_date) & panel.quotes["optionid"].isin(hedge_ids)].drop_duplicates("optionid").set_index("optionid").reindex(hedge_ids).reset_index()
    preview = src.scenarios_for_interval(panel, first_date, second_date, current_target, current_hedges)
    if np.asarray(preview.target_changes).shape != (8,):
        raise AssertionError("scenario source target change shape mismatch")
    if np.asarray(preview.hedge_changes).shape != (8, len(panel.hedges)):
        raise AssertionError("scenario source hedge change shape mismatch")
    bt = run_daily_backtest(panel, src, alpha_grid=np.array([0.01, 0.02]))
    write_json(out_dir / "self_check.json", {"shared_grid": summary, "gradient_matching": gm, "metrics": metrics, "scenario_target_shape": list(np.asarray(preview.target_changes).shape), "scenario_hedge_shape": list(np.asarray(preview.hedge_changes).shape), "backtest_rows": int(bt.results.shape[0]), "backtest_skipped_intervals": int(bt.skipped_interval_count)})
    print("VOLGAN_EXPERIMENT_SELF_CHECK=PASS")


def _repricing_vectorize_self_check() -> int:
    """Synthetic correctness + bench-equivalence check for the vectorized
    repricing helpers.  Compares the OLD scalar path (bs_price + interpolated_iv
    + Python loop) to the NEW vectorized path on a k=8 / 3 hedges / 1 target
    (straddle) synthetic panel.  Asserts max-abs diff < 1e-6 element-wise on
    every output array (next_target, next_hedges, DirectScenarioChanges fields).
    """
    rng = np.random.default_rng(42)
    nm, nt = 11, 9
    m_grid = np.linspace(0.7, 1.3, nm)
    tau_grid = np.linspace(0.05, 1.0, nt)
    k = 8
    # Smooth, positive surfaces.
    base = 0.15 + 0.1 * (m_grid[:, None] - 1.0) ** 2 + 0.05 * tau_grid[None, :]
    surfaces = np.repeat(base[None, :, :], k, axis=0) + rng.normal(0, 0.005, size=(k, nm, nt))
    surfaces = np.clip(surfaces, 0.01, None)
    s0 = 4000.0
    returns = rng.normal(0, 0.02, size=k)
    spots_next = s0 * np.exp(returns)
    # Target: straddle = 1 call + 1 put at strike s0.
    tgt_strikes = np.array([s0, s0], dtype=float)
    tgt_taus = np.array([0.5, 0.5], dtype=float)
    tgt_cp = np.array(["C", "P"], dtype="<U1")
    # Hedges: 1 call, 1 put, 1 underlying.
    hed_strikes = np.array([s0 * 1.05, s0 * 0.95, s0], dtype=float)
    hed_taus = np.array([0.3, 0.7, 0.0], dtype=float)
    hed_cp = np.array(["C", "P", "U"], dtype="<U1")

    # OLD scalar path -- mirrors the original double loop.
    def _old_reprice(strikes, taus, cp_arr):
        n = strikes.shape[0]
        out = np.zeros((k, n), dtype=float)
        for ki in range(k):
            spot = float(spots_next[ki])
            surf = surfaces[ki]
            for j in range(n):
                cp = str(cp_arr[j]).upper()
                if cp == "U":
                    out[ki, j] = spot
                    continue
                sigma = interpolated_iv(surf, m_grid, tau_grid, float(strikes[j]) / spot, float(taus[j]), floor_nonpositive=True)
                out[ki, j] = bs_price(spot, float(strikes[j]), float(taus[j]), sigma, cp)
        return out

    # NEW vectorized path.
    def _new_reprice(strikes, taus, cp_arr):
        n = strikes.shape[0]
        spot_kn = spots_next.reshape(k, 1)
        strike_kn = np.broadcast_to(strikes.reshape(1, n), (k, n))
        tau_kn = np.broadcast_to(taus.reshape(1, n), (k, n))
        cp_kn = np.broadcast_to(cp_arr.reshape(1, n), (k, n))
        moneyness_kn = strike_kn / spot_kn
        non_u = cp_kn != "U"
        iv_kn = _interpolated_iv_vec(surfaces, m_grid, tau_grid, moneyness_kn, tau_kn)
        iv_for_floor = np.where(non_u, iv_kn, np.nan)
        iv_floored = _apply_iv_floor(iv_for_floor, surfaces) if non_u.any() else iv_kn
        return _bs_price_vec(spot_kn, strike_kn, tau_kn, iv_floored, cp_kn)

    old_t = _old_reprice(tgt_strikes, tgt_taus, tgt_cp)
    new_t = _new_reprice(tgt_strikes, tgt_taus, tgt_cp)
    old_h = _old_reprice(hed_strikes, hed_taus, hed_cp)
    new_h = _new_reprice(hed_strikes, hed_taus, hed_cp)

    diff_t = float(np.max(np.abs(old_t - new_t)))
    diff_h = float(np.max(np.abs(old_h - new_h)))

    # DirectScenarioChanges-equivalent comparison.
    cur_tgt = np.array([10.0, 12.0])
    cur_hed = np.array([5.0, 6.0, s0])
    old_dV = old_t.sum(axis=1) - cur_tgt.sum()
    new_dV = new_t.sum(axis=1) - cur_tgt.sum()
    old_dH = old_h - cur_hed.reshape(1, -1)
    new_dH = new_h - cur_hed.reshape(1, -1)
    diff_dV = float(np.max(np.abs(old_dV - new_dV)))
    diff_dH = float(np.max(np.abs(old_dH - new_dH)))
    max_diff = max(diff_t, diff_h, diff_dV, diff_dH)
    print(f"vectorize_self_check max_abs_diff_target={diff_t:.3e} hedges={diff_h:.3e} dV={diff_dV:.3e} dH={diff_dH:.3e}", flush=True)
    assert max_diff < 1e-6, f"vectorized repricing diverges from scalar: {max_diff}"

    # Microbenchmark: k=1000, 10 contracts.
    import time
    rng2 = np.random.default_rng(0)
    kb = 1000
    nc = 10
    surfb = np.clip(0.15 + rng2.normal(0, 0.01, size=(kb, nm, nt)), 0.01, None)
    strikes_b = s0 * (0.9 + 0.2 * rng2.random(nc))
    taus_b = 0.05 + 0.95 * rng2.random(nc)
    cp_b = np.array(["C" if i % 2 == 0 else "P" for i in range(nc)], dtype="<U1")
    spots_b = s0 * np.exp(rng2.normal(0, 0.02, size=kb))

    t0 = time.perf_counter()
    out_old = np.zeros((kb, nc), dtype=float)
    for ki in range(kb):
        spot = float(spots_b[ki])
        surf = surfb[ki]
        for j in range(nc):
            sigma = interpolated_iv(surf, m_grid, tau_grid, float(strikes_b[j]) / spot, float(taus_b[j]), floor_nonpositive=True)
            out_old[ki, j] = bs_price(spot, float(strikes_b[j]), float(taus_b[j]), sigma, str(cp_b[j]))
    t_old = time.perf_counter() - t0

    t1 = time.perf_counter()
    spot_kn = spots_b.reshape(kb, 1)
    strike_kn = np.broadcast_to(strikes_b.reshape(1, nc), (kb, nc))
    tau_kn = np.broadcast_to(taus_b.reshape(1, nc), (kb, nc))
    cp_kn = np.broadcast_to(cp_b.reshape(1, nc), (kb, nc))
    moneyness_kn = strike_kn / spot_kn
    iv_kn = _interpolated_iv_vec(surfb, m_grid, tau_grid, moneyness_kn, tau_kn)
    iv_floored = _apply_iv_floor(iv_kn, surfb)
    out_new = _bs_price_vec(spot_kn, strike_kn, tau_kn, iv_floored, cp_kn)
    t_new = time.perf_counter() - t1
    bench_diff = float(np.max(np.abs(out_old - out_new)))
    speedup = t_old / max(t_new, 1e-12)
    print(f"vectorize_self_check bench_old={t_old:.3f}s bench_new={t_new:.3f}s speedup={speedup:.1f}x bench_max_abs_diff={bench_diff:.3e}", flush=True)
    assert bench_diff < 1e-6, f"bench-scale diverges: {bench_diff}"
    print("VECTORIZE_SELF_CHECK=PASS", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if getattr(args, "repricing_vectorize_self_check", False):
        return _repricing_vectorize_self_check()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    processed_dir = Path(args.processed_dir)
    data_dir = Path(args.data_dir)
    out_dir = make_output_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": vars(args),
        "device": str(device),
        "git_status": shell_capture(["git", "status", "--short", "--branch"]),
        "git_log": shell_capture(["git", "log", "--oneline", "--decorate", "-5"]),
        "torch_version": torch.__version__,
    }
    write_json(out_dir / "provenance.json", provenance)
    print(f"output_dir={out_dir}", flush=True)

    if args.stage == "self-check":
        self_check(processed_dir, data_dir, device, out_dir)
        return 0

    if args.stage == "report":
        input_dir = Path(args.report_input_dir) if args.report_input_dir else out_dir
        results_path = input_dir / "hedging_results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"missing hedging results for report stage: {results_path}")
        report_meta = write_paper_outputs(pd.read_csv(results_path), out_dir, {"report_input_dir": str(input_dir), "results_path": str(results_path)})
        print(f"paper_report_dir={out_dir / 'paper_reproduction'}")
        print(f"paper_report_status={report_meta.get('completeness_status')}")
        return 0

    train_results = []
    best_checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.stage in {"train", "end-to-end"}:
        for name in [x.strip() for x in args.configs.split(",") if x.strip()]:
            if name not in CONFIGS:
                raise ValueError(f"unknown config {name}; choices={sorted(CONFIGS)}")
            result = train_one(processed_dir, out_dir, CONFIGS[name], device)
            train_results.append(result)
        train_results = sorted(train_results, key=lambda row: float(row["metrics"]["selection_score"]))
        write_json(out_dir / "training_ranked.json", train_results)
        best_checkpoint = Path(train_results[0]["checkpoint"])
        print(f"best_checkpoint={best_checkpoint}", flush=True)
    if args.stage == "train":
        return 0

    if args.stage in {"hedge", "end-to-end"}:
        if best_checkpoint is None:
            raise ValueError("--checkpoint is required for --stage hedge")
        m0_values = [float(x) for x in args.hedge_m0.split(",") if x.strip()]
        run_hedging(best_checkpoint, processed_dir, data_dir, out_dir, device, args.hedge_samples, args.hedge_validation_samples, m0_values, args.hedge_start, args.hedge_end, args.hedge_max_periods, args.min_hedge_observed_frac, args.hedge_schedule, args.hedge_quote_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
