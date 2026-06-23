"""
VolGAN hedging backtest for one fixed m0 (ATM straddle by default).

Reproduces the pattern in Table 2 of Cont & Vuletić (2025):
  Unhedged → Delta → VolGAN LASSO  (monotonically improving tracking error)

Usage:
  python backtest_volgan.py \\
      --checkpoint /path/to/volgan_checkpoint.pt \\
      --prepared-dir data/volgan_prepared \\
      --data-dir data/VolGAN_optionmetrics_spx_20000103_20230228 \\
      --m0 1.0 \\
      --n-scenarios 1000 \\
      --n-val 100 \\
      --output results/table2_m1.csv
"""

import argparse
import json
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

# ─── VolGAN adapter shims (must happen before VolGAN import) ──────────────────
sys.path.insert(0, str(Path(__file__).parent / "../VolGAN"))
if "pandas_datareader" not in sys.modules:
    _stub = types.ModuleType("pandas_datareader")
    _stub.data = types.ModuleType("pandas_datareader.data")
    sys.modules["pandas_datareader"] = _stub
    sys.modules["pandas_datareader.data"] = _stub.data

import scipy as _scipy
if not hasattr(_scipy, "arange"):
    _scipy.arange = np.arange
    _scipy.array = np.array
    _scipy.exp = np.exp

import VolGAN as _VolGAN

# ─── Local modules ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from volgan_adapter import (
    GRID_ORDER, MONEYNESS_GRID, TAU_GRID, sample_scenarios, scenarios_to_solver_arrays,
    _bilinear_interp, _bs_price, _flat_to_surface,
)
from delta_surface import load_delta_surface, price_contracts as _ds_price, delta_vega_contracts as _ds_greeks

_hedging_spec = importlib.util.spec_from_file_location(
    "paper_protocol_hedging", Path(__file__).parent / "hedging.py"
)
if _hedging_spec is None or _hedging_spec.loader is None:
    raise ImportError("could not load local hedging.py")
_hedging = importlib.util.module_from_spec(_hedging_spec)
sys.modules[_hedging_spec.name] = _hedging
_hedging_spec.loader.exec_module(_hedging)
HedgePanel = _hedging.HedgePanel
build_instrument_panel = _hedging.build_instrument_panel
select_alpha_aic = _hedging.select_alpha_aic
solve_transaction_cost_lasso = _hedging.solve_transaction_cost_lasso
DATA_DIR = _hedging.DATA_DIR

RISK_FREE = 0.0  # paper uses r=0
UNDERLYING_OPTIONID = "UNDERLYING_SPX"
UNDERLYING_CP_FLAG = "U"


# ─── Market state helpers ─────────────────────────────────────────────────────

def build_state_lookup(prepared_dir: Path):
    """
    Load preprocessed surface + price data into fast date-keyed dicts.

    Returns
    -------
    dates        : list of pd.Timestamp (one per row)
    log_iv_rows  : [N, 80] log-IV surfaces
    closes       : [N] SPX closes
    log_rets     : [N] daily log-returns
    date_to_idx  : dict[pd.Timestamp -> int]
    """
    prepared_dir = Path(prepared_dir)
    grid_order = GRID_ORDER
    m_grid = MONEYNESS_GRID
    tau_grid = TAU_GRID

    if (prepared_dir / "surface_tensor.npz").exists():
        tensor = np.load(prepared_dir / "surface_tensor.npz")
        log_iv_tensor = np.asarray(tensor["log_iv"], dtype=float)
        if log_iv_tensor.ndim != 3:
            raise ValueError(f"log_iv tensor must be 3-dimensional, got {log_iv_tensor.shape}")
        log_iv_rows = log_iv_tensor.reshape(log_iv_tensor.shape[0], -1)
        dates = [pd.Timestamp(str(d)) for d in np.asarray(tensor["dates"]).astype(str)]
        log_rets = np.asarray(tensor["log_return"], dtype=float)
        prices_df = pd.read_csv(prepared_dir / "spx_daily.csv.gz", parse_dates=["date"])
        prices_df = prices_df[prices_df["date"].isin(dates)].set_index("date").loc[dates]
        closes = prices_df["spx_close"].values.astype(float)
        grid_path = prepared_dir / "grid_config.json"
        if grid_path.exists():
            grid = json.loads(grid_path.read_text())
            m_grid = np.asarray(grid.get("moneyness_grid", m_grid), dtype=float)
            tau_grid = np.asarray(grid.get("tau_grid", tau_grid), dtype=float)
            grid_order = str(grid.get("grid_order", grid_order))
    else:
        surfaces_df = pd.read_csv(prepared_dir / "surfaces_transform.csv", index_col=0)
        prices_df = pd.read_csv(prepared_dir / "spx_prices.csv", parse_dates=["date"])
        dates_df = pd.read_csv(prepared_dir / "dates.csv", parse_dates=["date"])

        raw_iv = surfaces_df.values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_iv_rows = np.log(np.clip(raw_iv, 1e-6, None))

        closes = prices_df["close"].values.astype(float)
        log_rets = prices_df["log_return"].values.astype(float)
        dates = [pd.Timestamp(d) for d in dates_df["date"]]

    date_to_idx = {d: i for i, d in enumerate(dates)}

    expected_points = len(m_grid) * len(tau_grid)
    if log_iv_rows.shape[1] != expected_points:
        raise ValueError(
            f"state surface width {log_iv_rows.shape[1]} does not match "
            f"grid {len(m_grid)}x{len(tau_grid)}={expected_points}"
        )

    return dates, log_iv_rows, closes, log_rets, date_to_idx, m_grid, tau_grid, grid_order


def get_day_state(date, date_to_idx, log_iv_rows, closes, log_rets):
    """
    Retrieve VolGAN conditioning state for a given trading date.

    Returns (log_iv_flat [80], spot, log_ret_tm1, log_ret_tm2, realised_vol) or None.
    """
    idx = date_to_idx.get(pd.Timestamp(date))
    if idx is None or idx < 22:
        return None
    log_iv_flat = log_iv_rows[idx]
    spot = closes[idx]
    r_tm1 = log_rets[idx - 1] if not np.isnan(log_rets[idx - 1]) else 0.0
    r_tm2 = log_rets[idx - 2] if not np.isnan(log_rets[idx - 2]) else 0.0
    rv = np.sqrt(252.0 / 21) * np.sqrt(np.nansum(log_rets[idx - 21 : idx] ** 2))
    return log_iv_flat, spot, r_tm1, r_tm2, rv


# ─── Instrument value helpers ─────────────────────────────────────────────────

def get_option_prices(quotes: pd.DataFrame, date: pd.Timestamp, optionids) -> np.ndarray | None:
    """
    Look up mid_price for a list of optionids on a given date.
    Returns [len(optionids)] array or None if any are missing.
    """
    day = quotes[quotes["date"] == date]
    prices = []
    for oid in optionids:
        row = day[day["optionid"] == oid]
        if row.empty:
            return None
        prices.append(float(row["mid_price"].iloc[0]))
    return np.array(prices)


def get_half_spreads(quotes: pd.DataFrame, date: pd.Timestamp, optionids) -> np.ndarray:
    """Look up half bid-ask spreads (transaction costs c_i) for each instrument."""
    day = quotes[quotes["date"] == date]
    costs = []
    for oid in optionids:
        row = day[day["optionid"] == oid]
        costs.append(float(row["half_spread"].iloc[0]) if not row.empty else 0.0)
    return np.array(costs)


def split_hedge_universe(panel: HedgePanel) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    """Return sorted hedge metadata, option rows, option indices, and underlying index."""

    hedges = panel.hedges.sort_values(["cp_flag", "strike"]).reset_index(drop=True)
    underlying_mask = (
        hedges["optionid"].map(lambda value: str(value) == UNDERLYING_OPTIONID)
        | hedges["cp_flag"].astype(str).eq(UNDERLYING_CP_FLAG)
        | hedges["role"].astype(str).eq("underlying")
    )
    underlying_indices = np.flatnonzero(underlying_mask.to_numpy(dtype=bool))
    if underlying_indices.size != 1:
        raise ValueError(f"VolGAN backtest requires exactly one underlying hedge row; found {underlying_indices.size}")
    option_indices = np.flatnonzero(~underlying_mask.to_numpy(dtype=bool))
    option_hedges = hedges.iloc[option_indices].reset_index(drop=True)
    return hedges, option_hedges, option_indices, int(underlying_indices[0])


def assemble_total_vector(
    n_hedges: int,
    option_indices: np.ndarray,
    option_values: np.ndarray,
    underlying_idx: int,
    spot: float,
) -> np.ndarray:
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
    changes = np.empty((len(spots_next), n_hedges), dtype=float)
    changes[:, option_indices] = np.asarray(option_changes, dtype=float)
    changes[:, underlying_idx] = np.asarray(spots_next, dtype=float) - float(spot_current)
    return changes


def assemble_total_greeks(
    n_hedges: int,
    option_indices: np.ndarray,
    option_deltas: np.ndarray,
    option_vegas: np.ndarray,
    underlying_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.zeros(n_hedges, dtype=float)
    vegas = np.zeros(n_hedges, dtype=float)
    deltas[option_indices] = np.asarray(option_deltas, dtype=float)
    vegas[option_indices] = np.asarray(option_vegas, dtype=float)
    deltas[underlying_idx] = 1.0
    return deltas, vegas


# ─── Delta baseline ───────────────────────────────────────────────────────────

def straddle_bs_delta(spot, strike, tau, sigma, r=0.0):
    """
    Delta of a long straddle (call + put) via Black-Scholes.
    = N(d1) + (N(d1) - 1) = 2*N(d1) - 1
    """
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return float(2.0 * norm.cdf(d1) - 1.0)


# ─── Contracts with updated tau ───────────────────────────────────────────────

def _set_tau(contracts, tau_val):
    """Return a copy of contracts DataFrame with tau column set to tau_val."""
    c = contracts.copy()
    c["tau"] = max(tau_val, 1.0 / 365)
    return c


def bs_price_from_surface(
    log_iv_flat: np.ndarray,
    spot: float,
    contracts,
    r: float = 0.0,
    m_grid: np.ndarray = MONEYNESS_GRID,
    tau_grid: np.ndarray = TAU_GRID,
    grid_order: str = GRID_ORDER,
) -> np.ndarray:
    """
    Price contracts using the VolGAN IV surface via bilinear interpolation + BS.

    log_iv_flat : [80] log-IV surface (tau-major flat: position i*10+j = tau_i, moneyness_j)
    spot        : scalar current spot price
    contracts   : DataFrame with cp_flag, strike, tau columns
    Returns     : [n_contracts] BS prices
    """
    strikes = contracts["strike"].values.astype(float)
    taus = contracts["tau"].values.astype(float)
    cp_flags = list(contracts["cp_flag"].values)

    iv_surface = np.exp(_flat_to_surface(log_iv_flat[None, :], m_grid, tau_grid, grid_order))
    m_query = (strikes / spot)[None, :]                              # [1, n_contracts]
    sigmas = _bilinear_interp(iv_surface, m_grid, tau_grid, m_query, taus)  # [1, n_contracts]
    prices = _bs_price(np.array([spot]), strikes, taus, sigmas, cp_flags, r)        # [1, n_contracts]
    return prices[0]  # [n_contracts]


# ─── Single window backtest ───────────────────────────────────────────────────

def run_one_window(
    panel: HedgePanel,
    gen: _VolGAN.Generator,
    state_lookup,
    n_scenarios: int,
    n_val: int,
    noise_dim: int,
    device: str,
    delta_surface_lookup: dict,
):
    """
    Run the VolGAN LASSO backtest + baselines for one hedging window.

    Returns a dict with keys "unhedged", "delta", "delta_vega", "volgan",
    each a list of daily tracking errors Z_t = V_t - Pi_t for t = 1..T.
    Returns None if data is too sparse to run.

    Realized P&L is marked using the OptionMetrics delta-grid surface
    (delta_surface_lookup), which is independent of the NW-smoothed training
    surface used by VolGAN.  Scenario generation for the LASSO still uses the
    NW surface so the optimization is unchanged.
    """
    dates, log_iv_rows, closes, log_rets, date_to_idx, m_grid, tau_grid, grid_order = state_lookup
    trading_dates = panel.trading_dates
    sorted_hedges, option_hedges, option_indices, underlying_idx = split_hedge_universe(panel)
    hedge_ids = list(sorted_hedges["optionid"])
    if panel.fixed_atm_optionid is None:
        return None
    atm_matches = sorted_hedges["optionid"].map(lambda value: str(value) == str(panel.fixed_atm_optionid)).to_numpy(dtype=bool)
    if int(atm_matches.sum()) != 1:
        return None
    atm_idx = int(np.flatnonzero(atm_matches)[0])

    # Contracts as DataFrames (tau will be updated at each step)
    target_contracts = panel.target.sort_values(["cp_flag", "strike"])[
        ["cp_flag", "strike", "ttm"]
    ].rename(columns={"ttm": "tau"}).reset_index(drop=True)
    hedge_contracts = option_hedges[
        ["cp_flag", "strike", "ttm"]
    ].rename(columns={"ttm": "tau"}).reset_index(drop=True)

    expiry = panel.expiry_date
    n_days = len(trading_dates)
    if n_days < 2:
        return None

    # ── t=0: NW surface prices for scenario conditioning and g0_scale ──
    t0_date = trading_dates[0]
    t0_state = get_day_state(t0_date, date_to_idx, log_iv_rows, closes, log_rets)
    if t0_state is None:
        return None
    log_iv_t0, spot_t0, r_tm1_t0, r_tm2_t0, rvol_t0 = t0_state

    tau_t0 = max((expiry - t0_date).days / 365, 1.0 / 365)
    tc_t0 = _set_tau(target_contracts, tau_t0)
    hc_t0 = _set_tau(hedge_contracts, tau_t0)

    # NW surface: used only for scenario generation and LASSO g0_scale
    t0_target_prices = bs_price_from_surface(
        log_iv_t0, spot_t0, tc_t0, r=RISK_FREE,
        m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
    )
    t0_hedge_prices_option = bs_price_from_surface(
        log_iv_t0, spot_t0, hc_t0, r=RISK_FREE,
        m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
    )
    t0_hedge_prices = assemble_total_vector(len(sorted_hedges), option_indices, t0_hedge_prices_option, underlying_idx, spot_t0)
    V0 = float(t0_target_prices.sum())   # g0_scale for LASSO; NW surface is fine here

    # Delta-grid surface: used for all realized P&L tracking
    t0_day_df = delta_surface_lookup.get(pd.Timestamp(t0_date))
    if t0_day_df is None:
        return None
    t0_target_prices_ds = _ds_price(t0_day_df, spot_t0, tc_t0, r=RISK_FREE)
    t0_hedge_prices_ds_option = _ds_price(t0_day_df, spot_t0, hc_t0, r=RISK_FREE)
    t0_hedge_prices_ds = assemble_total_vector(len(sorted_hedges), option_indices, t0_hedge_prices_ds_option, underlying_idx, spot_t0)
    V0_ds = float(t0_target_prices_ds.sum())
    if V0_ds <= 0:
        return None

    # ── AIC alpha selection at t=0 ──
    # Transaction costs: use t0 market half-spreads, fall back to 0 for missing optionids
    phi_zero = np.zeros(len(hedge_ids))
    c_t0 = get_half_spreads(panel.quotes, t0_date, hedge_ids)

    # N training scenarios
    spots_tr, iv_tr = sample_scenarios(
        gen, log_iv_t0, spot_t0, r_tm1_t0, r_tm2_t0, rvol_t0,
        N=n_scenarios, noise_dim=noise_dim, device=device,
        m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
    )
    dV_tr, dH_tr_option = scenarios_to_solver_arrays(
        spots_tr, iv_tr, spot_t0, tc_t0, hc_t0,
        t0_target_prices, t0_hedge_prices_option, r=RISK_FREE,
        m_grid=m_grid, tau_grid=tau_grid,
    )
    dH_tr = assemble_total_scenarios(len(sorted_hedges), option_indices, dH_tr_option, underlying_idx, spots_tr, spot_t0)
    # M validation scenarios
    spots_val, iv_val = sample_scenarios(
        gen, log_iv_t0, spot_t0, r_tm1_t0, r_tm2_t0, rvol_t0,
        N=n_val, noise_dim=noise_dim, device=device,
        m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
    )
    dV_val, dH_val_option = scenarios_to_solver_arrays(
        spots_val, iv_val, spot_t0, tc_t0, hc_t0,
        t0_target_prices, t0_hedge_prices_option, r=RISK_FREE,
        m_grid=m_grid, tau_grid=tau_grid,
    )
    dH_val = assemble_total_scenarios(len(sorted_hedges), option_indices, dH_val_option, underlying_idx, spots_val, spot_t0)

    alpha_best = select_alpha_aic(
        dV_tr, dH_tr, dV_val, dH_val,
        phi_prev=phi_zero, c_i=c_t0, g0_scale=V0,
    )

    # ── Rolling loop ──
    phi_volgan = phi_zero.copy()
    Pi_volgan  = V0_ds   # delta-grid initial value
    phi_delta  = 0.0     # scalar (units of underlying)
    Pi_delta   = V0_ds
    phi_vega_atm     = 0.0   # position in ATM hedge option (delta-vega)
    phi_delta_under  = 0.0   # position in underlying (delta-vega residual)
    Pi_dv      = V0_ds

    Z_volgan, Z_delta, Z_dv, Z_unhedged = [], [], [], []

    for step in range(n_days - 1):
        date_t = trading_dates[step]
        date_tp1 = trading_dates[step + 1]

        state_t = get_day_state(date_t, date_to_idx, log_iv_rows, closes, log_rets)
        state_tp1 = get_day_state(date_tp1, date_to_idx, log_iv_rows, closes, log_rets)
        if state_t is None or state_tp1 is None:
            break
        log_iv_t, spot_t, r_tm1_t, r_tm2_t, rvol_t = state_t
        log_iv_tp1, spot_tp1, _, _, _ = state_tp1

        tau_t = max((expiry - date_t).days / 365, 1.0 / 365)
        tau_tp1 = max((expiry - date_tp1).days / 365, 1.0 / 365)

        # ── Realized P&L prices via OptionMetrics delta-grid surface ──
        # Using the delta-grid surface (independent of VolGAN's NW training surface)
        # breaks the circularity that made tracking errors implausibly low.
        day_df_t   = delta_surface_lookup.get(pd.Timestamp(date_t))
        day_df_tp1 = delta_surface_lookup.get(pd.Timestamp(date_tp1))
        if day_df_t is None or day_df_tp1 is None:
            break

        tc_ds_t   = _set_tau(target_contracts, tau_t)
        hc_ds_t   = _set_tau(hedge_contracts,  tau_t)
        tc_ds_tp1 = _set_tau(target_contracts, tau_tp1)
        hc_ds_tp1 = _set_tau(hedge_contracts,  tau_tp1)

        prices_target_t   = _ds_price(day_df_t,   spot_t,   tc_ds_t,   r=RISK_FREE)
        prices_hedge_t_option = _ds_price(day_df_t,   spot_t,   hc_ds_t,   r=RISK_FREE)
        prices_target_tp1 = _ds_price(day_df_tp1, spot_tp1, tc_ds_tp1, r=RISK_FREE)
        prices_hedge_tp1_option = _ds_price(day_df_tp1, spot_tp1, hc_ds_tp1, r=RISK_FREE)
        prices_hedge_t = assemble_total_vector(len(sorted_hedges), option_indices, prices_hedge_t_option, underlying_idx, spot_t)
        prices_hedge_tp1 = assemble_total_vector(len(sorted_hedges), option_indices, prices_hedge_tp1_option, underlying_idx, spot_tp1)

        V_t   = float(prices_target_t.sum())
        V_tp1 = float(prices_target_tp1.sum())

        # Transaction costs fixed at t0 market levels (half bid-ask spread)
        c_t = c_t0

        # ── Unhedged ──
        Z_unhedged.append(V_tp1 - V0_ds)

        # ── Delta baseline ──
        # Greek computation via delta-grid surface at actual straddle moneyness
        tgt_deltas, _ = _ds_greeks(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        delta_t = float(tgt_deltas.sum())  # straddle delta = call_delta + put_delta

        phi_delta_new = delta_t
        trade_cost_delta = 0.0  # zero-cost underlying convention
        psi_delta = Pi_delta - phi_delta_new * spot_t - trade_cost_delta
        Pi_delta_new = phi_delta_new * spot_tp1 + psi_delta * (1 + RISK_FREE / 252)
        Z_delta.append(V_tp1 - Pi_delta_new)
        Pi_delta = Pi_delta_new
        phi_delta = phi_delta_new

        # ── Delta-vega baseline (paper §4.4) ──
        # phi_vega = straddle_vega / ATM_hedge_vega; residual delta via underlying
        tgt_deltas_dv, tgt_vegas_dv = _ds_greeks(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        hdg_deltas_option, hdg_vegas_option = _ds_greeks(day_df_t, spot_t, hc_ds_t, r=RISK_FREE)
        hdg_deltas_dv, hdg_vegas_dv = assemble_total_greeks(
            len(sorted_hedges), option_indices, hdg_deltas_option, hdg_vegas_option, underlying_idx
        )
        target_delta_dv = float(tgt_deltas_dv.sum())
        target_vega_dv  = float(tgt_vegas_dv.sum())
        kappa_h = float(hdg_vegas_dv[atm_idx])
        delta_h = float(hdg_deltas_dv[atm_idx])

        phi_vega_new   = target_vega_dv / kappa_h if abs(kappa_h) > 1e-12 else 0.0
        phi_delta_new  = target_delta_dv - phi_vega_new * delta_h

        trade_cost_dv = float(c_t0[atm_idx] * abs(phi_vega_new - phi_vega_atm))
        # Self-financing: subtract cost of NEW positions at current prices (Convention A,
        # matching LASSO).  Using old positions here would create a phantom gain/loss equal
        # to (phi_new - phi_old) * price_t — the source of the observed blow-up.
        psi_dv = Pi_dv - phi_vega_new * prices_hedge_t[atm_idx] - phi_delta_new * spot_t - trade_cost_dv
        Pi_dv_new = (phi_vega_new   * prices_hedge_tp1[atm_idx]
                     + phi_delta_new * spot_tp1
                     + psi_dv * (1 + RISK_FREE / 252))
        Z_dv.append(V_tp1 - Pi_dv_new)
        Pi_dv           = Pi_dv_new
        phi_vega_atm    = phi_vega_new
        phi_delta_under = phi_delta_new

        # ── VolGAN LASSO ──
        # Scenarios use NW surface (unchanged); realized P&L uses delta-grid prices above
        tc_nw = _set_tau(target_contracts, tau_tp1)
        hc_nw = _set_tau(hedge_contracts,  tau_tp1)

        # NW-based current prices for scenario diffs (separate from realized P&L)
        prices_target_t_nw = bs_price_from_surface(
            log_iv_t, spot_t, _set_tau(target_contracts, tau_t), r=RISK_FREE,
            m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
        )
        prices_hedge_t_nw_option = bs_price_from_surface(
            log_iv_t, spot_t, _set_tau(hedge_contracts,  tau_t), r=RISK_FREE,
            m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
        )
        prices_hedge_t_nw = assemble_total_vector(len(sorted_hedges), option_indices, prices_hedge_t_nw_option, underlying_idx, spot_t)

        spots_next, iv_next = sample_scenarios(
            gen, log_iv_t, spot_t, r_tm1_t, r_tm2_t, rvol_t,
            N=n_scenarios, noise_dim=noise_dim, device=device,
            m_grid=m_grid, tau_grid=tau_grid, grid_order=grid_order,
        )
        dV_t, dH_t_option = scenarios_to_solver_arrays(
            spots_next, iv_next, spot_t, tc_nw, hc_nw,
            prices_target_t_nw, prices_hedge_t_nw_option, r=RISK_FREE,
            m_grid=m_grid, tau_grid=tau_grid,
        )
        dH_t = assemble_total_scenarios(len(sorted_hedges), option_indices, dH_t_option, underlying_idx, spots_next, spot_t)

        result = solve_transaction_cost_lasso(
            dV_t, dH_t, phi_volgan, c_t, alpha=alpha_best, g0_scale=V0,
        )
        phi_new    = result.phi
        trade_cost = float(np.dot(c_t, np.abs(result.trade)))
        # Realized P&L uses delta-grid prices (not NW)
        psi        = Pi_volgan - float(np.dot(phi_new, prices_hedge_t)) - trade_cost
        Pi_volgan_new = float(np.dot(phi_new, prices_hedge_tp1)) + psi * (1 + RISK_FREE / 252)
        Z_volgan.append(V_tp1 - Pi_volgan_new)

        phi_volgan = phi_new
        Pi_volgan  = Pi_volgan_new

    if not Z_volgan:
        return None

    return {"unhedged": Z_unhedged, "delta": Z_delta, "delta_vega": Z_dv, "volgan": Z_volgan}


# ─── Evaluation ──────────────────────────────────────────────────────────────

def tracking_error_stats(Z: np.ndarray) -> dict:
    Z = np.asarray(Z)
    return {
        "n": len(Z),
        "mean": float(np.mean(Z)),
        "median": float(np.median(Z)),
        "std": float(np.std(Z)),
        "var_5pct": float(-np.percentile(Z, 5)),
        "var_2_5pct": float(-np.percentile(Z, 2.5)),
        "var_1pct": float(-np.percentile(Z, 1)),
    }


def load_volgan_generator(checkpoint: Path, device: str):
    """Load either current-schema or sibling VolGAN experiment checkpoints."""

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("gen_state", ckpt.get("gen_state_dict"))
    if state is None:
        raise KeyError("checkpoint must contain gen_state or gen_state_dict")

    cfg = ckpt.get("config", {})
    noise_dim = int(ckpt.get("noise_dim", cfg.get("noise_dim", 32)))
    hidden_dim = int(ckpt.get("hidden_dim", cfg.get("hidden_dim", state["linear1.bias"].shape[0])))
    cond_dim = ckpt.get("cond_dim")
    out_dim = ckpt.get("out_dim")
    if cond_dim is None:
        cond_dim = int(state["linear1.weight"].shape[1] - noise_dim)
    if out_dim is None:
        out_dim = int(state["linear3.weight"].shape[0])

    gen = _VolGAN.Generator(
        noise_dim=noise_dim,
        cond_dim=int(cond_dim),
        hidden_dim=hidden_dim,
        output_dim=int(out_dim),
    ).to(device)
    gen.load_state_dict(state)
    gen.eval()
    return gen, noise_dim, int(cond_dim), int(out_dim)


def print_table2(results: dict[str, list[float]]):
    header = f"{'Method':<18} {'N':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'VaR5%':>8} {'VaR2.5%':>9} {'VaR1%':>8}"
    print("\n" + header)
    print("-" * len(header))
    for method, Z in results.items():
        s = tracking_error_stats(Z)
        print(
            f"{method:<18} {s['n']:>6} {s['mean']:>8.3f} {s['median']:>8.3f} "
            f"{s['std']:>8.3f} {s['var_5pct']:>8.3f} {s['var_2_5pct']:>9.3f} {s['var_1pct']:>8.3f}"
        )


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to volgan_checkpoint.pt")
    parser.add_argument("--prepared-dir", type=Path,
                        default=Path("data/volgan_prepared"),
                        help="Directory with surfaces_transform.csv, spx_prices.csv, dates.csv")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/VolGAN_optionmetrics_spx_20000103_20230228"),
                        help="OptionMetrics raw data root (for build_instrument_panel)")
    parser.add_argument("--m0", type=float, default=1.0,
                        help="Straddle moneyness for a single run")
    parser.add_argument("--all-m0", action="store_true",
                        help="Pool all 6 paper moneyness values (overrides --m0)")
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test-start", default="2018-07-01")
    parser.add_argument("--test-end", default="2023-02-28")
    parser.add_argument("--max-windows", type=int, default=52)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--exclude-covid", action="store_true",
                        help="Exclude Covid-19 window (2020-02-13 to 2020-07-21)")
    args = parser.parse_args()

    # ── Load VolGAN ──
    print("Loading checkpoint ...")
    gen, noise_dim, cond_dim, out_dim = load_volgan_generator(args.checkpoint, args.device)
    print(f"  Generator dims: noise={noise_dim}, cond={cond_dim}, output={out_dim}")

    # ── Load preprocessed NW surface state ──
    print("Loading preprocessed surfaces ...")
    state_lookup = build_state_lookup(args.prepared_dir)
    _, log_iv_rows, _, _, _, m_grid, tau_grid, grid_order = state_lookup
    expected_cond = 3 + log_iv_rows.shape[1]
    expected_out = 1 + log_iv_rows.shape[1]
    if cond_dim != expected_cond or out_dim != expected_out:
        raise ValueError(
            f"checkpoint dims cond={cond_dim}, out={out_dim} do not match "
            f"prepared grid cond={expected_cond}, out={expected_out} "
            f"({len(m_grid)}x{len(tau_grid)}, order={grid_order})"
        )

    # ── Load OptionMetrics delta-grid surface for realized P&L marking ──
    test_start = pd.Timestamp(args.test_start)
    test_end   = pd.Timestamp(args.test_end)
    print("Loading OptionMetrics delta-grid surface ...")
    delta_surface_lookup = load_delta_surface(
        args.data_dir,
        start_year=test_start.year - 1,   # one year buffer for early windows
        end_year=test_end.year,
    )
    print(f"  Loaded {len(delta_surface_lookup)} daily surfaces "
          f"({min(delta_surface_lookup):%Y-%m-%d} to {max(delta_surface_lookup):%Y-%m-%d})")

    # ── Moneyness values to run ──
    # Paper §4 pools m0 ∈ {0.8, 0.85, 0.9, 0.95, 1.0, 1.05}; Table 2: n=1092=52×21
    M0_PAPER = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
    m0_values = M0_PAPER if args.all_m0 else [args.m0]

    # ── Monthly window candidates ──
    monthly_starts = pd.date_range(test_start, test_end, freq="MS")
    covid_start = pd.Timestamp("2020-02-13")
    covid_end   = pd.Timestamp("2020-07-21")

    # ── Run backtest ──
    results_all: dict[str, list[float]] = {
        "unhedged": [], "delta": [], "delta_vega": [], "volgan": [],
    }
    n_windows = 0

    for m0 in m0_values:
        if args.all_m0:
            print(f"\n{'─'*60}")
            print(f"m0 = {m0}")
        for candidate in monthly_starts:
            if not args.all_m0 and n_windows >= args.max_windows:
                break
            if args.exclude_covid and covid_start <= candidate <= covid_end:
                continue

            label = f"m0={m0} {candidate.date()}" if args.all_m0 else f"{candidate.date()}"
            print(f"  {label} ...", end=" ", flush=True)
            try:
                panel = build_instrument_panel(candidate, m0=m0, data_dir=args.data_dir)
            except Exception as e:
                print(f"SKIP (panel: {e})")
                continue

            window_results = run_one_window(
                panel, gen, state_lookup,
                n_scenarios=args.n_scenarios,
                n_val=args.n_val,
                noise_dim=noise_dim,
                device=args.device,
                delta_surface_lookup=delta_surface_lookup,
            )
            if window_results is None:
                print("SKIP (insufficient data)")
                continue

            for method in results_all:
                results_all[method].extend(window_results[method])

            n_days_done = len(window_results["volgan"])
            print(f"OK ({n_days_done} days, "
                  f"Z_volgan std={np.std(window_results['volgan']):.3f})")
            n_windows += 1

    print(f"\n{'='*60}")
    print(f"Total windows: {n_windows}, total observations: {len(results_all['volgan'])}")

    if not results_all["volgan"]:
        print("No results to report.")
        return

    print_table2(results_all)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for method, Z in results_all.items():
            s = tracking_error_stats(Z)
            rows.append({"method": method, **s})
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
