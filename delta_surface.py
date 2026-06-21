"""OptionMetrics delta-grid surface loader and Black-Scholes utilities.

Provides realized P&L marking that is independent of the NW-smoothed training
surface used by VolGAN.  The vsurfd files (vol_surface_delta_grid/) have one row
per (date, days, delta, cp_flag) and cover all trading days without gaps.

Interpolation is bilinear in (impl_moneyness = impl_strike/spot, days) space.
Boundary behaviour: clamp to grid edge rather than extrapolate.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_delta_surface(
    data_dir: Path,
    start_year: int,
    end_year: int,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load vsurfd files and return {date → DataFrame} lookup.

    Each DataFrame has columns: days, delta, cp_flag, impl_volatility,
    impl_strike, impl_premium (and secid / dispersion which are ignored).
    """
    frames = []
    for year in range(start_year, end_year + 1):
        path = data_dir / "vol_surface_delta_grid" / f"spx_vsurfd_{year}.csv.gz"
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["date"]))
    if not frames:
        raise FileNotFoundError(
            f"No vsurfd files found under {data_dir}/vol_surface_delta_grid/"
        )
    df = pd.concat(frames, ignore_index=True)
    return {
        pd.Timestamp(d): sub.reset_index(drop=True)
        for d, sub in df.groupby("date")
    }


# ─── Interpolation ────────────────────────────────────────────────────────────

def iv_from_delta_surface(
    day_df: pd.DataFrame,
    spot: float,
    cp_flag: str,
    strike: float,
    tau_days: float,
) -> float:
    """Bilinear interpolation of IV in (impl_moneyness, days) space.

    impl_moneyness is computed on-the-fly as impl_strike / spot so no
    pre-computation per date is needed.  Clamps to grid boundary rather than
    extrapolating.  Returns at least 1e-6 to avoid downstream division errors.
    """
    side = day_df[day_df["cp_flag"] == cp_flag]
    if side.empty:
        return 0.20  # safe fallback
    m_target = strike / spot
    days_arr = np.sort(side["days"].unique())

    # Bracket tau in the days dimension
    below = days_arr[days_arr <= tau_days]
    above = days_arr[days_arr >= tau_days]
    d_lo = float(below[-1]) if len(below) else float(days_arr[0])
    d_hi = float(above[0])  if len(above) else float(days_arr[-1])

    def _mono_interp(d_val: float) -> float:
        s = side[side["days"] == d_val]
        m_vals = (s["impl_strike"].values / spot).astype(float)
        iv_vals = s["impl_volatility"].values.astype(float)
        order = np.argsort(m_vals)
        return float(np.interp(m_target, m_vals[order], iv_vals[order]))

    iv_lo = _mono_interp(d_lo)
    if d_lo == d_hi:
        return max(iv_lo, 1e-6)
    iv_hi = _mono_interp(d_hi)
    w = (tau_days - d_lo) / (d_hi - d_lo)
    return max((1.0 - w) * iv_lo + w * iv_hi, 1e-6)


# ─── Black-Scholes scalars ────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(
    spot: float, strike: float, tau: float, sigma: float,
    cp_flag: str, r: float = 0.0,
) -> float:
    if tau <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 0.0
    sqt = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqt)
    d2 = d1 - sigma * sqt
    disc = math.exp(-r * tau)
    if cp_flag == "C":
        return spot * _normal_cdf(d1) - strike * disc * _normal_cdf(d2)
    return strike * disc * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def _bs_delta(
    spot: float, strike: float, tau: float, sigma: float,
    cp_flag: str, r: float = 0.0,
) -> float:
    if tau <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 0.0
    d1 = (
        math.log(spot / strike) + (r + 0.5 * sigma * sigma) * tau
    ) / (sigma * math.sqrt(tau))
    nd1 = _normal_cdf(d1)
    return nd1 if cp_flag == "C" else nd1 - 1.0


def _bs_vega(
    spot: float, strike: float, tau: float, sigma: float, r: float = 0.0,
) -> float:
    if tau <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 0.0
    sqt = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqt)
    return spot * sqt * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)


# ─── Contract-level pricing ───────────────────────────────────────────────────

def price_contracts(
    day_df: pd.DataFrame,
    spot: float,
    contracts: pd.DataFrame,
    r: float = 0.0,
) -> np.ndarray:
    """Price each contract via delta-grid IV + Black-Scholes.

    contracts: DataFrame with columns cp_flag, strike, tau (tau in years).
    Returns array of shape [n_contracts].
    """
    prices = np.empty(len(contracts), dtype=float)
    for j, (_, row) in enumerate(contracts.iterrows()):
        sigma = iv_from_delta_surface(
            day_df, spot, str(row["cp_flag"]),
            float(row["strike"]), float(row["tau"]) * 365.0,
        )
        prices[j] = _bs_price(
            spot, float(row["strike"]), float(row["tau"]),
            sigma, str(row["cp_flag"]), r,
        )
    return prices


def delta_vega_contracts(
    day_df: pd.DataFrame,
    spot: float,
    contracts: pd.DataFrame,
    r: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta, vega) arrays for each contract using delta-grid IV."""
    deltas = np.empty(len(contracts), dtype=float)
    vegas  = np.empty(len(contracts), dtype=float)
    for j, (_, row) in enumerate(contracts.iterrows()):
        sigma = iv_from_delta_surface(
            day_df, spot, str(row["cp_flag"]),
            float(row["strike"]), float(row["tau"]) * 365.0,
        )
        deltas[j] = _bs_delta(
            spot, float(row["strike"]), float(row["tau"]),
            sigma, str(row["cp_flag"]), r,
        )
        vegas[j] = _bs_vega(
            spot, float(row["strike"]), float(row["tau"]), sigma, r,
        )
    return deltas, vegas
