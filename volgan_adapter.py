"""Adapter from a trained VolGAN generator to the hedging pipeline.

Takes the VolGAN generator and the current market state, draws N one-step-ahead
scenarios, and returns solver-ready arrays for the LASSO solver. Black-Scholes
pricing is vectorized over the N scenarios.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

# VolGAN import with compatibility shims
sys.path.insert(0, str(Path(__file__).parent.parent / "VolGAN"))

import types as _types, sys as _sys
if "pandas_datareader" not in _sys.modules:
    _stub = _types.ModuleType("pandas_datareader")
    _stub.data = _types.ModuleType("pandas_datareader.data")
    _sys.modules["pandas_datareader"] = _stub
    _sys.modules["pandas_datareader.data"] = _stub.data

import scipy as _scipy
if not hasattr(_scipy, "arange"):
    _scipy.arange = np.arange
    _scipy.array = np.array
    _scipy.exp = np.exp

import VolGAN as _VolGAN  # noqa: E402 (after shims)

# Columns expected from hedging.py contracts DataFrames
_CP = "cp_flag"
_STRIKE = "strike"
_TAU = "tau"


# Grid

MONEYNESS_GRID = np.linspace(0.6, 1.4, 10)
TAU_DAYS = np.array([7, 14, 30, 60, 91, 182, 273, 365])
TAU_GRID = TAU_DAYS / 365.0


# Core math

def _bilinear_interp(surfaces: np.ndarray, m_grid: np.ndarray, tau_grid: np.ndarray,
                     m_query: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a regular (moneyness x tau) grid.

    surfaces is [N, n_m, n_tau] (one per scenario), m_query is [N, n_c] and
    tau_query is [n_c] (shared across scenarios). Returns [N, n_c].
    """
    N, n_m, n_tau = surfaces.shape
    n_c = m_query.shape[1]
    result = np.empty((N, n_c), dtype=float)
    k_idx = np.arange(N)

    for j in range(n_c):
        # tau cell (scalar; same for all scenarios)
        tau_j = float(np.clip(tau_query[j], tau_grid[0], tau_grid[-1]))
        t = np.searchsorted(tau_grid, tau_j) - 1
        t = int(np.clip(t, 0, n_tau - 2))
        wt2 = (tau_j - tau_grid[t]) / (tau_grid[t + 1] - tau_grid[t])
        wt1 = 1.0 - wt2

        # moneyness cell (vector; varies by scenario)
        m_j = np.clip(m_query[:, j], m_grid[0], m_grid[-1])
        m_idx = np.clip(np.searchsorted(m_grid, m_j) - 1, 0, n_m - 2)
        dm = m_grid[m_idx + 1] - m_grid[m_idx]
        wm2 = (m_j - m_grid[m_idx]) / dm
        wm1 = 1.0 - wm2

        result[:, j] = (
            wm1 * wt1 * surfaces[k_idx, m_idx, t]
            + wm1 * wt2 * surfaces[k_idx, m_idx, t + 1]
            + wm2 * wt1 * surfaces[k_idx, m_idx + 1, t]
            + wm2 * wt2 * surfaces[k_idx, m_idx + 1, t + 1]
        )

    return result


def _bs_price(spots: np.ndarray, strikes: np.ndarray, taus: np.ndarray,
              sigmas: np.ndarray, cp_flags: list[str], r: float = 0.0) -> np.ndarray:
    """Vectorized Black-Scholes pricing.

    spots is [N], strikes/taus/cp_flags are [n_c], sigmas is [N, n_c]. Returns
    [N, n_c] option prices; cp_flags entries are 'C' or 'P'.
    """
    S = spots[:, None]               # [N, 1]
    K = np.asarray(strikes)[None, :]  # [1, n_c]
    tau = np.asarray(taus)[None, :]   # [1, n_c]
    sigma = np.clip(sigmas, 1e-6, 10.0)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    disc = np.exp(-r * tau)

    is_call = np.array([c.upper() == "C" for c in cp_flags])[None, :]  # [1, n_c]
    call_price = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    put_price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


# Conditioning

def build_condition(log_iv_flat: np.ndarray, log_ret_tm1: float,
                    log_ret_tm2: float, realised_vol: float) -> np.ndarray:
    """Build the conditioning vector a_t consumed by VolGAN.

    Mirrors DataPreprocesssing() in VolGAN.py:
    [R_{t-1}*sqrt(252), R_{t-2}*sqrt(252), gamma_{t-1}, log_iv_{t-1}], where
    gamma is the 21-day realised vol and log_iv_flat is [80], tau-major. The two
    log-returns are raw daily values, annualized here.
    """
    return np.concatenate([
        [np.sqrt(252) * log_ret_tm1],
        [np.sqrt(252) * log_ret_tm2],
        [realised_vol],
        log_iv_flat,
    ])


# Main sampling function

def sample_scenarios(
    gen: _VolGAN.Generator,
    log_iv_flat: np.ndarray,
    spot: float,
    log_ret_tm1: float,
    log_ret_tm2: float,
    realised_vol: float,
    N: int = 1000,
    noise_dim: int = 32,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Draw N one-step-ahead scenarios from VolGAN.

    log_iv_flat is the [80] log current IV surface (tau-major); noise_dim must
    match training. Returns (spots_next [N], iv_next [N, 10, 8]), where iv_next
    is absolute IV, not log.
    """
    cond = build_condition(log_iv_flat, log_ret_tm1, log_ret_tm2, realised_vol)
    cond_t = torch.from_numpy(cond).float().to(device).unsqueeze(0).expand(N, -1)
    noise = torch.randn(N, noise_dim, device=device)

    gen.eval()
    with torch.no_grad():
        fake = gen(noise, cond_t).cpu().numpy()  # [N, 81]

    # fake[:,0] is annualized log-return; de-annualize to get daily
    log_ret_next = fake[:, 0] / np.sqrt(252)           # [N]
    spots_next = spot * np.exp(log_ret_next)            # [N]

    # IV surface: add increment to current log-IV, then exponentiate
    log_iv_next_flat = log_iv_flat[None, :] + fake[:, 1:]  # [N, 80]
    # Unflatten: [N, 80] → [N, 8 (tau), 10 (m)] → [N, 10, 8]
    iv_next = np.exp(log_iv_next_flat.reshape(N, 8, 10).transpose(0, 2, 1))

    return spots_next, iv_next


# Scenario to solver arrays

def scenarios_to_solver_arrays(
    spots_next: np.ndarray,
    iv_next: np.ndarray,
    spot_current: float,
    target_contracts,
    hedge_contracts,
    current_target_values: np.ndarray,
    current_hedge_values: np.ndarray,
    r: float = 0.0,
    m_grid: np.ndarray = MONEYNESS_GRID,
    tau_grid: np.ndarray = TAU_GRID,
):
    """Price target and hedge contracts under each scenario into solver arrays.

    iv_next is [N, 10, 8] absolute IV; target/hedge contracts are DataFrames with
    cp_flag, strike, tau columns. Returns (target_changes [N], hedge_changes
    [N, n_h]) as scenario P&L of the portfolio and each hedging instrument.
    """
    N = len(spots_next)
    all_contracts = list(target_contracts.iterrows()) + list(hedge_contracts.iterrows())
    cp_flags = [str(c[_CP]).upper() for _, c in all_contracts]
    strikes = np.array([float(c[_STRIKE]) for _, c in all_contracts])
    taus = np.array([float(c[_TAU]) for _, c in all_contracts])

    # Moneyness = strike / spot_next for each (scenario, contract)
    m_query = (strikes[None, :] / spots_next[:, None])  # [N, n_contracts]

    sigmas = _bilinear_interp(iv_next, m_grid, tau_grid, m_query, taus)  # [N, n_contracts]
    prices_next = _bs_price(spots_next, strikes, taus, sigmas, cp_flags, r)  # [N, n_contracts]

    n_t = len(target_contracts)
    # Portfolio value = sum of target instrument prices (long straddle: call + put)
    target_next = prices_next[:, :n_t].sum(axis=1)          # [N]
    hedge_next = prices_next[:, n_t:]                        # [N, n_h]

    target_changes = target_next - float(current_target_values.sum())
    hedge_changes = hedge_next - current_hedge_values[None, :]

    return target_changes, hedge_changes
