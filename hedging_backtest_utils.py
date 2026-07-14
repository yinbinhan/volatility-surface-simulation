"""Model-neutral utilities for volatility-surface hedging backtests."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from hedging import HedgePanel, UNDERLYING_CP_FLAG, UNDERLYING_OPTIONID


MONEYNESS_GRID = np.array(
    [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
)
TAU_GRID = np.array(
    [1 / 365, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1.0]
)
GRID_ORDER = "m_major_tau_minor"


def _flat_to_surface(
    flat: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    grid_order: str = GRID_ORDER,
) -> np.ndarray:
    """Unflatten surface vectors to arrays with shape [N, n_m, n_tau]."""

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


def _bilinear_interp(
    surfaces: np.ndarray,
    m_grid: np.ndarray,
    tau_grid: np.ndarray,
    m_query: np.ndarray,
    tau_query: np.ndarray,
) -> np.ndarray:
    """Interpolate scenario surfaces on a regular moneyness-maturity grid."""

    n_scenarios, n_m, n_tau = surfaces.shape
    n_contracts = m_query.shape[1]
    result = np.empty((n_scenarios, n_contracts), dtype=float)
    scenario_idx = np.arange(n_scenarios)

    for contract_idx in range(n_contracts):
        tau_value = float(
            np.clip(tau_query[contract_idx], tau_grid[0], tau_grid[-1])
        )
        tau_idx = int(
            np.clip(np.searchsorted(tau_grid, tau_value) - 1, 0, n_tau - 2)
        )
        tau_weight_high = (tau_value - tau_grid[tau_idx]) / (
            tau_grid[tau_idx + 1] - tau_grid[tau_idx]
        )
        tau_weight_low = 1.0 - tau_weight_high

        m_value = np.clip(
            m_query[:, contract_idx], m_grid[0], m_grid[-1]
        )
        m_idx = np.clip(np.searchsorted(m_grid, m_value) - 1, 0, n_m - 2)
        m_width = m_grid[m_idx + 1] - m_grid[m_idx]
        m_weight_high = (m_value - m_grid[m_idx]) / m_width
        m_weight_low = 1.0 - m_weight_high

        result[:, contract_idx] = (
            m_weight_low
            * tau_weight_low
            * surfaces[scenario_idx, m_idx, tau_idx]
            + m_weight_low
            * tau_weight_high
            * surfaces[scenario_idx, m_idx, tau_idx + 1]
            + m_weight_high
            * tau_weight_low
            * surfaces[scenario_idx, m_idx + 1, tau_idx]
            + m_weight_high
            * tau_weight_high
            * surfaces[scenario_idx, m_idx + 1, tau_idx + 1]
        )

    return result


def _bs_price(
    spots: np.ndarray,
    strikes: np.ndarray,
    taus: np.ndarray,
    sigmas: np.ndarray,
    cp_flags: list[str],
    r: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-Scholes prices for scenario-contract pairs."""

    spot = np.asarray(spots, dtype=float)[:, None]
    strike = np.asarray(strikes, dtype=float)[None, :]
    tau = np.asarray(taus, dtype=float)[None, :]
    sigma = np.clip(np.asarray(sigmas, dtype=float), 1e-6, 10.0)

    sqrt_tau = np.sqrt(tau)
    d1 = (
        np.log(spot / strike) + (r + 0.5 * sigma**2) * tau
    ) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    discount = np.exp(-r * tau)

    is_call = np.array(
        [flag.upper() == "C" for flag in cp_flags], dtype=bool
    )[None, :]
    call_price = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    put_price = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


def build_state_lookup(prepared_dir: Path):
    """Load preprocessed surfaces and SPX prices into date-indexed arrays."""

    prepared_dir = Path(prepared_dir)
    grid_order = GRID_ORDER
    m_grid = MONEYNESS_GRID
    tau_grid = TAU_GRID

    if (prepared_dir / "surface_tensor.npz").exists():
        tensor = np.load(prepared_dir / "surface_tensor.npz")
        log_iv_tensor = np.asarray(tensor["log_iv"], dtype=float)
        if log_iv_tensor.ndim != 3:
            raise ValueError(
                "log_iv tensor must be 3-dimensional, "
                f"got {log_iv_tensor.shape}"
            )
        log_iv_rows = log_iv_tensor.reshape(log_iv_tensor.shape[0], -1)
        dates = [
            pd.Timestamp(str(value))
            for value in np.asarray(tensor["dates"]).astype(str)
        ]
        log_rets = np.asarray(tensor["log_return"], dtype=float)
        prices_df = pd.read_csv(
            prepared_dir / "spx_daily.csv.gz", parse_dates=["date"]
        )
        prices_df = (
            prices_df[prices_df["date"].isin(dates)]
            .set_index("date")
            .loc[dates]
        )
        closes = prices_df["spx_close"].to_numpy(dtype=float)
        grid_path = prepared_dir / "grid_config.json"
        if grid_path.exists():
            grid = json.loads(grid_path.read_text())
            m_grid = np.asarray(
                grid.get("moneyness_grid", m_grid), dtype=float
            )
            tau_grid = np.asarray(grid.get("tau_grid", tau_grid), dtype=float)
            grid_order = str(grid.get("grid_order", grid_order))
    else:
        surfaces_df = pd.read_csv(
            prepared_dir / "surfaces_transform.csv", index_col=0
        )
        prices_df = pd.read_csv(
            prepared_dir / "spx_prices.csv", parse_dates=["date"]
        )
        dates_df = pd.read_csv(
            prepared_dir / "dates.csv", parse_dates=["date"]
        )

        raw_iv = surfaces_df.to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_iv_rows = np.log(np.clip(raw_iv, 1e-6, None))

        closes = prices_df["close"].to_numpy(dtype=float)
        log_rets = prices_df["log_return"].to_numpy(dtype=float)
        dates = [pd.Timestamp(value) for value in dates_df["date"]]

    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    expected_points = len(m_grid) * len(tau_grid)
    if log_iv_rows.shape[1] != expected_points:
        raise ValueError(
            f"state surface width {log_iv_rows.shape[1]} does not match "
            f"grid {len(m_grid)}x{len(tau_grid)}={expected_points}"
        )

    return (
        dates,
        log_iv_rows,
        closes,
        log_rets,
        date_to_idx,
        m_grid,
        tau_grid,
        grid_order,
    )


def get_day_state(date, date_to_idx, log_iv_rows, closes, log_rets):
    """Return the surface, spot, lagged returns, and realized volatility."""

    idx = date_to_idx.get(pd.Timestamp(date))
    if idx is None or idx < 22:
        return None
    log_iv_flat = log_iv_rows[idx]
    spot = closes[idx]
    return_lag_1 = (
        log_rets[idx - 1] if not np.isnan(log_rets[idx - 1]) else 0.0
    )
    return_lag_2 = (
        log_rets[idx - 2] if not np.isnan(log_rets[idx - 2]) else 0.0
    )
    realized_volatility = np.sqrt(252.0 / 21) * np.sqrt(
        np.nansum(log_rets[idx - 21 : idx] ** 2)
    )
    return (
        log_iv_flat,
        spot,
        return_lag_1,
        return_lag_2,
        realized_volatility,
    )


def get_half_spreads(
    quotes: pd.DataFrame,
    date: pd.Timestamp,
    optionids,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Return observed half spreads, using opening spreads when unavailable."""

    day = quotes[quotes["date"] == date]
    costs = []
    fallback_arr = (
        None if fallback is None else np.asarray(fallback, dtype=float)
    )
    for idx, optionid in enumerate(optionids):
        row = day[day["optionid"] == optionid]
        if not row.empty:
            value = float(row["half_spread"].iloc[0])
            if np.isfinite(value):
                costs.append(value)
                continue
        costs.append(
            float(fallback_arr[idx]) if fallback_arr is not None else 0.0
        )
    return np.asarray(costs, dtype=float)


def split_hedge_universe(
    panel: HedgePanel,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    """Separate option hedges from the single underlying hedge row."""

    hedges = panel.hedges.sort_values(
        ["cp_flag", "strike"]
    ).reset_index(drop=True)
    underlying_mask = (
        hedges["optionid"].map(
            lambda value: str(value) == UNDERLYING_OPTIONID
        )
        | hedges["cp_flag"].astype(str).eq(UNDERLYING_CP_FLAG)
        | hedges["role"].astype(str).eq("underlying")
    )
    underlying_indices = np.flatnonzero(
        underlying_mask.to_numpy(dtype=bool)
    )
    if underlying_indices.size != 1:
        raise ValueError(
            "hedging backtest requires exactly one underlying hedge row; "
            f"found {underlying_indices.size}"
        )
    option_indices = np.flatnonzero(
        ~underlying_mask.to_numpy(dtype=bool)
    )
    option_hedges = hedges.iloc[option_indices].reset_index(drop=True)
    return (
        hedges,
        option_hedges,
        option_indices,
        int(underlying_indices[0]),
    )


def assemble_total_vector(
    n_hedges: int,
    option_indices: np.ndarray,
    option_values: np.ndarray,
    underlying_idx: int,
    spot: float,
) -> np.ndarray:
    """Combine option values and the underlying value in hedge-row order."""

    values = np.empty(n_hedges, dtype=float)
    values[option_indices] = np.asarray(option_values, dtype=float)
    values[underlying_idx] = float(spot)
    return values


def assemble_total_scenarios(
    n_hedges: int,
    option_indices: np.ndarray,
    option_changes: np.ndarray,
    underlying_idx: int,
    spots_next: np.ndarray,
    spot_current: float,
) -> np.ndarray:
    """Combine option and underlying scenario changes in hedge-row order."""

    changes = np.empty((len(spots_next), n_hedges), dtype=float)
    changes[:, option_indices] = np.asarray(option_changes, dtype=float)
    changes[:, underlying_idx] = (
        np.asarray(spots_next, dtype=float) - float(spot_current)
    )
    return changes


def assemble_total_greeks(
    n_hedges: int,
    option_indices: np.ndarray,
    option_deltas: np.ndarray,
    option_vegas: np.ndarray,
    underlying_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine option Greeks with the underlying delta in hedge-row order."""

    deltas = np.zeros(n_hedges, dtype=float)
    vegas = np.zeros(n_hedges, dtype=float)
    deltas[option_indices] = np.asarray(option_deltas, dtype=float)
    vegas[option_indices] = np.asarray(option_vegas, dtype=float)
    deltas[underlying_idx] = 1.0
    return deltas, vegas


def set_contract_tau(contracts: pd.DataFrame, tau_value: float) -> pd.DataFrame:
    """Return contracts with their remaining maturity set to tau_value."""

    updated = contracts.copy()
    updated["tau"] = max(tau_value, 1.0 / 365)
    return updated


def bs_price_from_surface(
    log_iv_flat: np.ndarray,
    spot: float,
    contracts: pd.DataFrame,
    r: float = 0.0,
    m_grid: np.ndarray = MONEYNESS_GRID,
    tau_grid: np.ndarray = TAU_GRID,
    grid_order: str = GRID_ORDER,
) -> np.ndarray:
    """Price contracts from a log-IV surface using interpolation and BS."""

    strikes = contracts["strike"].to_numpy(dtype=float)
    taus = contracts["tau"].to_numpy(dtype=float)
    cp_flags = [str(value) for value in contracts["cp_flag"]]

    iv_surface = np.exp(
        _flat_to_surface(
            np.asarray(log_iv_flat, dtype=float)[None, :],
            m_grid,
            tau_grid,
            grid_order,
        )
    )
    m_query = (strikes / float(spot))[None, :]
    sigmas = _bilinear_interp(
        iv_surface, m_grid, tau_grid, m_query, taus
    )
    return _bs_price(
        np.asarray([spot], dtype=float),
        strikes,
        taus,
        sigmas,
        cp_flags,
        r,
    )[0]


def intrinsic_values(
    contracts: pd.DataFrame, spot: float
) -> np.ndarray:
    """Return exact call or put intrinsic values at settlement."""

    strikes = contracts["strike"].to_numpy(dtype=float)
    is_call = (
        contracts["cp_flag"]
        .astype(str)
        .str.upper()
        .str.startswith("C")
        .to_numpy()
    )
    call_values = np.maximum(float(spot) - strikes, 0.0)
    put_values = np.maximum(strikes - float(spot), 0.0)
    return np.where(is_call, call_values, put_values)


def scenarios_to_solver_arrays(
    spots_next: np.ndarray,
    iv_next: np.ndarray,
    spot_current: float,
    target_contracts: pd.DataFrame,
    hedge_contracts: pd.DataFrame,
    current_target_values: np.ndarray,
    current_hedge_values: np.ndarray,
    r: float = 0.0,
    m_grid: np.ndarray = MONEYNESS_GRID,
    tau_grid: np.ndarray = TAU_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Reprice contracts under surface scenarios and return P&L changes."""

    spots_next = np.asarray(spots_next, dtype=float)
    iv_next = np.asarray(iv_next, dtype=float)
    all_contracts = pd.concat(
        [target_contracts, hedge_contracts], ignore_index=True
    )
    cp_flags = [str(value).upper() for value in all_contracts["cp_flag"]]
    strikes = all_contracts["strike"].to_numpy(dtype=float)
    taus = all_contracts["tau"].to_numpy(dtype=float)

    m_query = strikes[None, :] / spots_next[:, None]
    sigmas = _bilinear_interp(
        iv_next, m_grid, tau_grid, m_query, taus
    )
    prices_next = _bs_price(
        spots_next, strikes, taus, sigmas, cp_flags, r
    )

    n_targets = len(target_contracts)
    target_next = prices_next[:, :n_targets].sum(axis=1)
    hedge_next = prices_next[:, n_targets:]
    target_changes = target_next - float(
        np.asarray(current_target_values, dtype=float).sum()
    )
    hedge_changes = hedge_next - np.asarray(
        current_hedge_values, dtype=float
    )[None, :]
    return target_changes, hedge_changes


def tracking_error_stats(values: np.ndarray) -> dict[str, float | int]:
    """Return descriptive tracking-error statistics."""

    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "var_5pct": float(-np.percentile(values, 5)),
        "var_2_5pct": float(-np.percentile(values, 2.5)),
        "var_1pct": float(-np.percentile(values, 1)),
    }


def print_tracking_error_table(
    results: dict[str, list[float]],
) -> None:
    """Print descriptive tracking-error statistics by method."""

    header = (
        f"{'Method':<18} {'N':>6} {'Mean':>8} {'Median':>8} "
        f"{'Std':>8} {'VaR5%':>8} {'VaR2.5%':>9} {'VaR1%':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for method, values in results.items():
        stats = tracking_error_stats(values)
        print(
            f"{method:<18} {stats['n']:>6} {stats['mean']:>8.3f} "
            f"{stats['median']:>8.3f} {stats['std']:>8.3f} "
            f"{stats['var_5pct']:>8.3f} "
            f"{stats['var_2_5pct']:>9.3f} "
            f"{stats['var_1pct']:>8.3f}"
        )
