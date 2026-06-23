"""
VolGAN → hedging pipeline adapter.

Takes a trained VolGAN generator and the current market state, generates N
one-step-ahead scenarios, and returns SolverScenarioArrays ready for the LASSO solver.

All BS pricing is vectorized over N scenarios via numpy; no Python loops over scenarios.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

# ─── VolGAN import with compatibility shims ───────────────────────────────────
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


# ─── Grid ─────────────────────────────────────────────────────────────────────

MONEYNESS_GRID = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4])
TAU_GRID = np.array([1 / 365, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1.0])
GRID_ORDER = "m_major_tau_minor"


# ─── Core math ────────────────────────────────────────────────────────────────

def _flat_to_surface(
    flat: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    grid_order: str = GRID_ORDER,
) -> np.ndarray:
    """Unflatten VolGAN vectors to [N, n_m, n_tau]."""
    flat = np.asarray(flat, dtype=float)
    n_m = len(m_grid)
    n_tau = len(tau_grid)
    expected = n_m * n_tau
    if flat.shape[-1] != expected:
        raise ValueError(f"surface has {flat.shape[-1]} points, expected {expected}")
    if grid_order == "m_major_tau_minor":
        return flat.reshape(flat.shape[0], n_m, n_tau)
    if grid_order == "tau_major_m_minor":
        return flat.reshape(flat.shape[0], n_tau, n_m).transpose(0, 2, 1)
    raise ValueError(f"unsupported grid_order={grid_order!r}")

def _bilinear_interp(surfaces: np.ndarray, m_grid: np.ndarray, tau_grid: np.ndarray,
                     m_query: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation on a regular (moneyness × tau) grid.

    Parameters
    ----------
    surfaces  : [N, n_m, n_tau] — one surface per scenario
    m_grid    : [n_m] sorted moneyness values
    tau_grid  : [n_tau] sorted tau values
    m_query   : [N, n_c] — moneyness query per (scenario, contract)
    tau_query : [n_c]    — tau query per contract (same across scenarios)

    Returns
    -------
    [N, n_c] interpolated values
    """
    N, n_m, n_tau = surfaces.shape
    n_c = m_query.shape[1]
    result = np.empty((N, n_c), dtype=float)
    k_idx = np.arange(N)

    for j in range(n_c):
        # ── tau cell (scalar; same for all scenarios) ──
        tau_j = float(np.clip(tau_query[j], tau_grid[0], tau_grid[-1]))
        t = np.searchsorted(tau_grid, tau_j) - 1
        t = int(np.clip(t, 0, n_tau - 2))
        wt2 = (tau_j - tau_grid[t]) / (tau_grid[t + 1] - tau_grid[t])
        wt1 = 1.0 - wt2

        # ── moneyness cell (vector; varies by scenario) ──
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
    """
    Vectorized Black-Scholes pricing.

    Parameters
    ----------
    spots   : [N] — spot price per scenario
    strikes : [n_c] — strike per contract
    taus    : [n_c] — time to maturity per contract (years)
    sigmas  : [N, n_c] — implied vol per (scenario, contract)
    cp_flags: [n_c] — 'C' or 'P'

    Returns
    -------
    [N, n_c] option prices
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


# ─── Conditioning ─────────────────────────────────────────────────────────────

def build_condition(log_iv_flat: np.ndarray, log_ret_tm1: float,
                    log_ret_tm2: float, realised_vol: float) -> np.ndarray:
    """
    Build the conditioning vector a_t consumed by VolGAN.

    DataPreprocesssing() in VolGAN.py constructs:
        condition = [R_{t-1}*sqrt(252), R_{t-2}*sqrt(252), gamma_{t-1}, log_iv_{t-1}]
    where gamma is the 21-day realised vol.

    Parameters
    ----------
    log_iv_flat   : [n_m * n_tau] — log of the current IV surface
    log_ret_tm1   : raw daily log-return at t-1 (will be annualized internally)
    log_ret_tm2   : raw daily log-return at t-2 (will be annualized internally)
    realised_vol  : 21-day realised vol at t-1 = sqrt(252/21 * sum(r_{t-i}^2))

    Returns
    -------
    [3 + n_m * n_tau] — conditioning vector
    """
    return np.concatenate([
        [np.sqrt(252) * log_ret_tm1],
        [np.sqrt(252) * log_ret_tm2],
        [realised_vol],
        log_iv_flat,
    ])


# ─── Main sampling function ───────────────────────────────────────────────────

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
    m_grid: np.ndarray = MONEYNESS_GRID,
    tau_grid: np.ndarray = TAU_GRID,
    grid_order: str = GRID_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw N one-step-ahead scenarios from VolGAN.

    Parameters
    ----------
    gen          : trained VolGAN Generator
    log_iv_flat  : [n_m * n_tau] log current IV surface
    spot         : current SPX spot price
    log_ret_tm1  : raw log-return at t-1
    log_ret_tm2  : raw log-return at t-2
    realised_vol : annualized 21-day realised vol
    N            : number of scenarios
    noise_dim    : generator noise dimension (must match training)
    device       : "cpu" or "cuda"

    Returns
    -------
    spots_next : [N] — next-day spot prices
    iv_next    : [N, n_m, n_tau] — next-day IV surfaces (absolute values, not log)
    """
    cond = build_condition(log_iv_flat, log_ret_tm1, log_ret_tm2, realised_vol)
    cond_t = torch.from_numpy(cond).float().to(device).unsqueeze(0).expand(N, -1)
    noise = torch.randn(N, noise_dim, device=device)

    gen.eval()
    with torch.no_grad():
        fake = gen(noise, cond_t).cpu().numpy()

    surface_points = len(m_grid) * len(tau_grid)
    if fake.shape[1] != surface_points + 1:
        raise ValueError(
            f"VolGAN output dimension {fake.shape[1]} is incompatible with "
            f"{surface_points} grid points plus one return increment"
        )

    # fake[:,0] is annualized log-return; de-annualize to get daily
    log_ret_next = fake[:, 0] / np.sqrt(252)           # [N]
    spots_next = spot * np.exp(log_ret_next)            # [N]

    # IV surface: add increment to current log-IV, then exponentiate
    log_iv_next_flat = log_iv_flat[None, :] + fake[:, 1:]
    iv_next = np.exp(_flat_to_surface(log_iv_next_flat, m_grid, tau_grid, grid_order))

    return spots_next, iv_next


# ─── Scenario → SolverScenarioArrays ─────────────────────────────────────────

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
    """
    Price target and hedge contracts under each scenario, then compute P&L changes.

    Parameters
    ----------
    spots_next            : [N]
    iv_next               : [N, 10, 8] — absolute IV surfaces
    spot_current          : scalar
    target_contracts      : pd.DataFrame with cp_flag, strike, tau columns
    hedge_contracts       : pd.DataFrame with cp_flag, strike, tau columns
    current_target_values : [n_target] current BS/market prices
    current_hedge_values  : [n_hedge] current BS/market prices
    r                     : risk-free rate

    Returns
    -------
    target_changes : [N]       — scenario P&L of the portfolio
    hedge_changes  : [N, n_h]  — scenario P&L of each hedging instrument
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
