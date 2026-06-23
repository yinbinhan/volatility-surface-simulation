"""Diffusion-model hedging backtest (Cont & Vuletić 2025).

The diffusion analogue of backtest_volgan.py. Everything downstream of the
scenario generator is reused unchanged: panel construction, the LASSO + AIC
solver, realized-P&L marking off the OptionMetrics delta grid, and the delta /
delta-vega baselines. Only the generator differs: the diffusion model conditions
on the last 21 days of 3-channel 11x9 surfaces (diffusion_state.py) rather than
VolGAN's single-day surface.

Scenarios depend on the date but not on m0, so each date is sampled once and
cached to --scenario-cache for reuse across the m0 passes and the AIC validation
draw. The output is a per-observation CSV that eval_hedging.py turns into Table 2
and the figures.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from diffusion_hedging_adapter import (
    DiffusionHedgingAdapter,
    MONEYNESS_GRID,
    TAU_GRID,
    sample_scenarios,
    scenarios_to_solver_arrays,
    _bilinear_interp,
    _bs_price,
)
from diffusion_state import DiffusionState, CONDITIONING_LENGTH
from hedging import (
    HedgePanel,
    build_instrument_panel,
    select_alpha_aic,
    solve_transaction_cost_lasso,
    DATA_DIR,
)
from delta_surface import (
    load_delta_surface,
    price_contracts as _ds_price,
    delta_vega_contracts as _ds_greeks,
)

RISK_FREE = 0.0  # paper uses r=0

# Delta-vega baseline: minimum hedge-option vega (as a fraction of the straddle
# vega) for the vega leg to be rebalanced. Cont & Vuletić §4.4 note a low-vega
# instrument gives an unstable ϕ1; this bounds |ϕ1| ≤ 1/VEGA_FLOOR_FRAC and tames
# the near-expiry κ_H→0 blow-up of the fixed-K=S0 option (body is unaffected).
VEGA_FLOOR_FRAC = 0.05  # → |ϕ1| ≤ 20

# Paper m0 grid (Cont & Vuletić Tables 5/7, Figs 7-13). Differs from the grid
# hard-coded in backtest_volgan.py; see CLAUDE.md "m0-grid correction".
M0_PAPER = [0.75, 0.80, 0.90, 1.10, 1.20, 1.25]


# Transaction-cost / contract helpers (shared with backtest_volgan)

def get_half_spreads(quotes: pd.DataFrame, date: pd.Timestamp, optionids) -> np.ndarray:
    """Half bid-ask spreads (transaction costs c_i) for each instrument at date."""
    day = quotes[quotes["date"] == date]
    costs = []
    for oid in optionids:
        row = day[day["optionid"] == oid]
        costs.append(float(row["half_spread"].iloc[0]) if not row.empty else 0.0)
    return np.array(costs)


def _set_tau(contracts, tau_val):
    """Copy of contracts with the tau column set to tau_val (>= 1 day)."""
    c = contracts.copy()
    c["tau"] = max(tau_val, 1.0 / 365)
    return c


def bs_price_from_iv_surface(iv_surface: np.ndarray, spot: float,
                             contracts, r: float = 0.0) -> np.ndarray:
    """Price contracts on a date-t 11x9 IV surface (bilinear interp + BS).

    Shares _bilinear_interp / _bs_price with scenarios_to_solver_arrays so the
    current prices V_t/H_t are on the same footing as the scenario prices.
    """
    strikes = contracts["strike"].values.astype(float)
    taus = contracts["tau"].values.astype(float)
    cp_flags = [str(x).upper() for x in contracts["cp_flag"].values]
    iv = iv_surface[None, :, :]                          # [1, 11, 9]
    m_query = (strikes / spot)[None, :]                  # [1, n]
    sigmas = _bilinear_interp(iv, MONEYNESS_GRID, TAU_GRID, m_query, taus)  # [1, n]
    prices = _bs_price(np.array([spot]), strikes, taus, sigmas, cp_flags, r)  # [1, n]
    return prices[0]


# Scenario cache (m0-independent, keyed by date)

def get_scenario_pool(
    adapter: DiffusionHedgingAdapter,
    diff_state: DiffusionState,
    date_t,
    n_total: int,
    cache_dir: Path | None,
):
    """Return (spots, iv) scenarios for one day ahead of date_t, or None.

    Cached per date because scenarios are identical across m0 values. The
    per-date seed is deterministic so cached and freshly sampled draws agree.
    Returns None when there is insufficient conditioning history.
    """
    ts = pd.Timestamp(date_t)
    cache_file = cache_dir / f"{ts.date()}.npz" if cache_dir is not None else None

    if cache_file is not None and cache_file.exists():
        z = np.load(cache_file)
        if z["spots"].shape[0] >= n_total:
            return z["spots"][:n_total].copy(), z["iv"][:n_total].copy()

    hist = diff_state.get_conditioning_history(date_t, n_cond=CONDITIONING_LENGTH)
    if hist is None:
        return None
    history, spot_t = hist

    # Deterministic per-date seed → reproducible across runs and cache rebuilds.
    idx = diff_state.date_to_idx[ts]
    torch.manual_seed(20260603 + idx)

    spots, iv = sample_scenarios(adapter, history, spot_t, N=n_total)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            spots=spots.astype(np.float32),
            iv=iv.astype(np.float32),
            date=str(ts.date()),
        )
    return spots, iv


# Single-window backtest

def run_one_window(
    panel: HedgePanel,
    adapter: DiffusionHedgingAdapter,
    diff_state: DiffusionState,
    n_scenarios: int,
    n_val: int,
    delta_surface_lookup: dict,
    cache_dir: Path | None,
    fixed_alpha: float | None = None,
    include_underlying: bool = True,
    underlying_cost_bps: float = 0.5,
):
    """Run diffusion-LASSO + baselines for one hedging window.

    Returns dict with keys "unhedged", "delta", "delta_vega", "diffusion"
    (each a list of daily tracking errors Z_t = V_t - Pi_t) plus "dates"
    (the date_{t+1} for each observation).  Returns None if data is too sparse.
    """
    trading_dates = panel.trading_dates
    hedge_ids = list(panel.hedges.sort_values(["cp_flag", "strike"])["optionid"])

    target_contracts = panel.target.sort_values(["cp_flag", "strike"])[
        ["cp_flag", "strike", "ttm"]
    ].rename(columns={"ttm": "tau"}).reset_index(drop=True)
    hedge_contracts = panel.hedges.sort_values(["cp_flag", "strike"])[
        ["cp_flag", "strike", "ttm"]
    ].rename(columns={"ttm": "tau"}).reset_index(drop=True)

    expiry = panel.expiry_date
    n_days = len(trading_dates)
    if n_days < 2:
        return None

    # t=0
    t0_date = trading_dates[0]
    spot_t0 = diff_state.get_spot(t0_date)
    iv_surf_t0 = diff_state.get_iv_surface(t0_date)
    if spot_t0 is None or iv_surf_t0 is None or not diff_state.has_history(t0_date):
        return None

    tau_t0 = max((expiry - t0_date).days / 365, 1.0 / 365)
    tc_t0 = _set_tau(target_contracts, tau_t0)
    hc_t0 = _set_tau(hedge_contracts, tau_t0)

    # Diffusion-surface current prices (for scenario diffs and g0_scale)
    t0_target_prices = bs_price_from_iv_surface(iv_surf_t0, spot_t0, tc_t0, r=RISK_FREE)
    t0_hedge_prices  = bs_price_from_iv_surface(iv_surf_t0, spot_t0, hc_t0, r=RISK_FREE)
    V0 = float(t0_target_prices.sum())

    # Delta-grid surface: all realized P&L tracking (independent of training surface)
    t0_day_df = delta_surface_lookup.get(pd.Timestamp(t0_date))
    if t0_day_df is None:
        return None
    t0_target_prices_ds = _ds_price(t0_day_df, spot_t0, tc_t0, r=RISK_FREE)
    V0_ds = float(t0_target_prices_ds.sum())
    if V0_ds <= 0:
        return None

    # Paper §4.4: fix the delta-vega hedge option at window initiation to the
    # candidate nearest ATM (K = S0) and hold it to expiry; only the ratio
    # rebalances daily (no live-ATM reselection — see backtest_volgan.py note).
    _hc_strikes = hedge_contracts["strike"].values
    atm_idx0 = int(np.argmin(np.abs(_hc_strikes / spot_t0 - 1.0)))

    # AIC alpha selection at t=0 (train + independent validation draw)
    pool = get_scenario_pool(adapter, diff_state, t0_date, n_scenarios + n_val, cache_dir)
    if pool is None:
        return None
    spots_pool, iv_pool = pool
    spots_tr, iv_tr = spots_pool[:n_scenarios], iv_pool[:n_scenarios]
    spots_val, iv_val = spots_pool[n_scenarios:n_scenarios + n_val], iv_pool[n_scenarios:n_scenarios + n_val]

    c_t0 = get_half_spreads(panel.quotes, t0_date, hedge_ids)
    # Diffusion LASSO instrument set. Optionally prepend the underlying as
    # instrument 0 (delta=1, vega=0, low cost) so the regression can hedge delta
    # without buying vega — Cont & Vuletić's paper-faithful set (§4.3). The
    # baselines below are untouched and keep using the options-only c_t0 / arrays.
    if include_underlying:
        u_cost0 = underlying_cost_bps / 1e4 * spot_t0
        c_diff0 = np.concatenate([[u_cost0], c_t0])
    else:
        c_diff0 = c_t0
    phi_zero = np.zeros(len(c_diff0))   # diffusion hedge vector (len n_h [+1])

    if fixed_alpha is not None:
        alpha_best = float(fixed_alpha)   # Fig 14 regularization-robustness runs
    else:
        dV_tr, dH_tr = scenarios_to_solver_arrays(
            spots_tr, iv_tr, spot_t0, tc_t0, hc_t0,
            t0_target_prices, t0_hedge_prices, r=RISK_FREE,
            include_underlying=include_underlying,
        )
        dV_val, dH_val = scenarios_to_solver_arrays(
            spots_val, iv_val, spot_t0, tc_t0, hc_t0,
            t0_target_prices, t0_hedge_prices, r=RISK_FREE,
            include_underlying=include_underlying,
        )
        alpha_best = select_alpha_aic(
            dV_tr, dH_tr, dV_val, dH_val,
            phi_prev=phi_zero, c_i=c_diff0, g0_scale=V0,
        )

    # Rolling loop
    phi_diff = phi_zero.copy()
    Pi_diff  = V0_ds
    phi_delta = 0.0
    Pi_delta  = V0_ds
    phi_vega_atm    = 0.0
    Pi_dv     = V0_ds

    Z_diff, Z_delta, Z_dv, Z_unhedged, obs_dates = [], [], [], [], []
    # Diagnostics for paper Figs 3/4/5 + Tables 1/3/4 (all from the diffusion hedge)
    diag_Vt, diag_ninstr = [], []
    diag_straddle_delta, diag_straddle_vega = [], []
    diag_hedged_delta, diag_hedged_vega = [], []

    for step in range(n_days - 1):
        date_t = trading_dates[step]
        date_tp1 = trading_dates[step + 1]

        spot_t = diff_state.get_spot(date_t)
        spot_tp1 = diff_state.get_spot(date_tp1)
        iv_surf_t = diff_state.get_iv_surface(date_t)
        if spot_t is None or spot_tp1 is None or iv_surf_t is None:
            break
        if not diff_state.has_history(date_t):
            break

        tau_t = max((expiry - date_t).days / 365, 1.0 / 365)
        tau_tp1 = max((expiry - date_tp1).days / 365, 1.0 / 365)

        # Realized P&L via OptionMetrics delta-grid surface
        day_df_t   = delta_surface_lookup.get(pd.Timestamp(date_t))
        day_df_tp1 = delta_surface_lookup.get(pd.Timestamp(date_tp1))
        if day_df_t is None or day_df_tp1 is None:
            break

        tc_ds_t   = _set_tau(target_contracts, tau_t)
        hc_ds_t   = _set_tau(hedge_contracts,  tau_t)
        tc_ds_tp1 = _set_tau(target_contracts, tau_tp1)
        hc_ds_tp1 = _set_tau(hedge_contracts,  tau_tp1)

        prices_target_t   = _ds_price(day_df_t,   spot_t,   tc_ds_t,   r=RISK_FREE)
        prices_hedge_t    = _ds_price(day_df_t,   spot_t,   hc_ds_t,   r=RISK_FREE)
        prices_target_tp1 = _ds_price(day_df_tp1, spot_tp1, tc_ds_tp1, r=RISK_FREE)
        prices_hedge_tp1  = _ds_price(day_df_tp1, spot_tp1, hc_ds_tp1, r=RISK_FREE)

        V_tp1 = float(prices_target_tp1.sum())
        # Transaction costs fixed at t0 market levels: options use c_t0; the
        # diffusion hedge uses c_diff0 (= c_t0, optionally with the underlying
        # half-spread prepended as instrument 0).

        # Unhedged
        Z_unhedged.append(V_tp1 - V0_ds)

        # Delta baseline
        tgt_deltas, _ = _ds_greeks(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        delta_t = float(tgt_deltas.sum())
        phi_delta = delta_t                       # set hedge at t (fixes the off-by-one)
        psi_delta = Pi_delta - phi_delta * spot_t
        Pi_delta_new = phi_delta * spot_tp1 + psi_delta * (1 + RISK_FREE / 252)
        Z_delta.append(V_tp1 - Pi_delta_new)
        Pi_delta = Pi_delta_new

        # Delta-vega baseline (paper §4.4: fixed K=S0 option, ratio rebalanced).
        # Single option fixed at window init (atm_idx0), held to expiry; only the
        # ratio rebalances. phi_vega = kappa_V/kappa_H, phi_delta = Delta_V - phi_vega*Delta_H.
        tgt_deltas_dv, tgt_vegas_dv = _ds_greeks(day_df_t, spot_t, tc_ds_t, r=RISK_FREE)
        hdg_deltas_dv, hdg_vegas_dv = _ds_greeks(day_df_t, spot_t, hc_ds_t, r=RISK_FREE)
        target_delta_dv = float(tgt_deltas_dv.sum())
        target_vega_dv  = float(tgt_vegas_dv.sum())
        kappa_h = float(hdg_vegas_dv[atm_idx0])
        delta_h = float(hdg_deltas_dv[atm_idx0])
        # Vega-floor guard (paper §4.4: low-vega instrument → unstable ϕ1). Rebalance
        # the vega leg only while the fixed option carries a usable vega; else carry
        # the prior ratio. Bounds |ϕ1| ≤ 1/VEGA_FLOOR_FRAC, removing the near-expiry
        # κ_H→0 blow-up while leaving the (paper-matching) body untouched.
        if abs(kappa_h) > VEGA_FLOOR_FRAC * abs(target_vega_dv) and abs(kappa_h) > 1e-8:
            phi_vega_new = target_vega_dv / kappa_h
        else:
            phi_vega_new = phi_vega_atm
        phi_delta_new = target_delta_dv - phi_vega_new * delta_h
        # Same option every day → spread cost only on the change in its position.
        trade_cost_dv = float(c_t0[atm_idx0] * abs(phi_vega_new - phi_vega_atm))
        psi_dv = Pi_dv - phi_vega_new * prices_hedge_t[atm_idx0] - phi_delta_new * spot_t - trade_cost_dv
        Pi_dv_new = (phi_vega_new * prices_hedge_tp1[atm_idx0]
                     + phi_delta_new * spot_tp1
                     + psi_dv * (1 + RISK_FREE / 252))
        Z_dv.append(V_tp1 - Pi_dv_new)
        Pi_dv = Pi_dv_new
        phi_vega_atm = phi_vega_new

        # Diffusion LASSO.
        # Current prices on the date-t diffusion surface (for scenario diffs);
        # scenario prices use next-day tau (time decay), mirroring backtest_volgan.
        prices_target_t_dfm = bs_price_from_iv_surface(
            iv_surf_t, spot_t, _set_tau(target_contracts, tau_t), r=RISK_FREE)
        prices_hedge_t_dfm = bs_price_from_iv_surface(
            iv_surf_t, spot_t, _set_tau(hedge_contracts, tau_t), r=RISK_FREE)

        pool_t = get_scenario_pool(adapter, diff_state, date_t, n_scenarios, cache_dir)
        if pool_t is None:
            break
        spots_next, iv_next = pool_t

        dV_t, dH_t = scenarios_to_solver_arrays(
            spots_next, iv_next, spot_t,
            _set_tau(target_contracts, tau_tp1), _set_tau(hedge_contracts, tau_tp1),
            prices_target_t_dfm, prices_hedge_t_dfm, r=RISK_FREE,
            include_underlying=include_underlying,
        )
        # Realized-P&L prices for the diffusion hedge roll-forward. With the
        # underlying as instrument 0, prepend its realized values (current spot_t,
        # next spot_tp1) so the self-financing update picks up the underlying leg
        # exactly as the delta baseline does; options keep their delta-grid prices.
        if include_underlying:
            prices_hedge_t_roll   = np.concatenate([[spot_t],   prices_hedge_t])
            prices_hedge_tp1_roll = np.concatenate([[spot_tp1], prices_hedge_tp1])
        else:
            prices_hedge_t_roll   = prices_hedge_t
            prices_hedge_tp1_roll = prices_hedge_tp1

        result = solve_transaction_cost_lasso(
            dV_t, dH_t, phi_diff, c_diff0, alpha=alpha_best, g0_scale=V0,
        )
        phi_new = result.phi
        trade_cost = float(np.dot(c_diff0, np.abs(result.trade)))
        psi = Pi_diff - float(np.dot(phi_new, prices_hedge_t_roll)) - trade_cost
        Pi_diff_new = float(np.dot(phi_new, prices_hedge_tp1_roll)) + psi * (1 + RISK_FREE / 252)
        Z_diff.append(V_tp1 - Pi_diff_new)
        phi_diff = phi_new
        Pi_diff = Pi_diff_new

        obs_dates.append(pd.Timestamp(date_tp1))

        # Diagnostics for the diffusion hedge (Figs 3/4/5, Tables 1/3/4).
        # Greeks reuse the delta-vega block's date-t arrays. The hedged-position
        # greek = straddle greek − Σ φ_i·(hedge-instrument greek). With the
        # underlying as instrument 0 it contributes delta=1, vega=0; cash is zero.
        # (Options-only mode: the underlying is absent, so hedged_delta need not ~0.)
        if include_underlying:
            hdg_deltas_full = np.concatenate([[1.0], hdg_deltas_dv])
            hdg_vegas_full  = np.concatenate([[0.0], hdg_vegas_dv])
        else:
            hdg_deltas_full = hdg_deltas_dv
            hdg_vegas_full  = hdg_vegas_dv
        diag_Vt.append(V_tp1)
        diag_ninstr.append(int(np.sum(np.abs(phi_new) > 1e-8)))
        diag_straddle_delta.append(target_delta_dv)
        diag_straddle_vega.append(target_vega_dv)
        diag_hedged_delta.append(target_delta_dv - float(np.dot(phi_new, hdg_deltas_full)))
        diag_hedged_vega.append(target_vega_dv - float(np.dot(phi_new, hdg_vegas_full)))

    if not Z_diff:
        return None
    return {
        "unhedged": Z_unhedged, "delta": Z_delta, "delta_vega": Z_dv,
        "diffusion": Z_diff, "dates": obs_dates,
        "alpha": [float(alpha_best)] * len(Z_diff),
        "V_t": diag_Vt, "n_instruments": diag_ninstr,
        "straddle_delta": diag_straddle_delta, "straddle_vega": diag_straddle_vega,
        "hedged_delta": diag_hedged_delta, "hedged_vega": diag_hedged_vega,
    }


# Evaluation helpers

def tracking_error_stats(Z) -> dict:
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


def print_table2(results: dict[str, list]):
    header = (f"{'Method':<18} {'N':>6} {'Mean':>8} {'Median':>8} {'Std':>8} "
              f"{'VaR5%':>8} {'VaR2.5%':>9} {'VaR1%':>8}")
    print("\n" + header)
    print("-" * len(header))
    for method, Z in results.items():
        if not len(Z):
            continue
        s = tracking_error_stats(Z)
        print(f"{method:<18} {s['n']:>6} {s['mean']:>8.3f} {s['median']:>8.3f} "
              f"{s['std']:>8.3f} {s['var_5pct']:>8.3f} {s['var_2_5pct']:>9.3f} {s['var_1pct']:>8.3f}")


# Entry point

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-npy", type=Path,
                        default=Path("data/shared_grid_11x9/shared_grid_22d_logiv_call_return.npy"),
                        help="Training .npy for shape + normalization stats")
    parser.add_argument("--surface-tensor", type=Path,
                        default=Path("data/processed_shared_grid_11x9/surface_tensor.npz"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/VolGAN_optionmetrics_spx_20000103_20230228"))
    parser.add_argument("--m0", type=float, default=0.90)
    parser.add_argument("--all-m0", action="store_true",
                        help="Pool all paper moneyness values (overrides --m0)")
    parser.add_argument("--n-scenarios", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--device", default=None, help="cuda/mps/cpu (auto if unset)")
    parser.add_argument("--sampling-timesteps", type=int, default=50,
                        help="DDIM steps (50 default); 200 = faithful DDPM")
    parser.add_argument("--test-start", default="2018-07-01")
    parser.add_argument("--test-end", default="2023-02-28")
    parser.add_argument("--max-windows", type=int, default=52)
    parser.add_argument("--scenario-cache", type=Path, default=Path("scenarios/diff_ddim50_ep3000"))
    parser.add_argument("--obs-output", type=Path, default=Path("results/diffusion_obs.csv"))
    parser.add_argument("--fixed-alpha", type=float, default=None,
                        help="If set, skip AIC and use this constant LASSO alpha "
                             "(for the Fig 14 regularization-robustness runs).")
    parser.add_argument("--include-underlying", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Add the underlying as LASSO instrument 0 (paper-faithful "
                             "set, Cont & Vuletić §4.3 'underlying always selected'; "
                             "DEFAULT on). Use --no-include-underlying for the legacy "
                             "options-only set. See CLAUDE.md finding #5.")
    parser.add_argument("--underlying-cost-bps", type=float, default=0.5,
                        help="Underlying half-spread transaction cost in bps of spot "
                             "(only used with --include-underlying; default 0.5).")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")

    instr_mode = (f"underlying+options (cost {args.underlying_cost_bps}bps)"
                  if args.include_underlying else "options-only")
    print(f"Device: {args.device} | sampling_timesteps={args.sampling_timesteps} "
          f"| LASSO instruments: {instr_mode}")
    print("Building diffusion adapter ...")
    adapter = DiffusionHedgingAdapter(
        checkpoint_path=args.checkpoint,
        data_path=args.data_npy,
        n_scenarios=args.n_scenarios,
        device=args.device,
        sampling_timesteps=args.sampling_timesteps,
    )

    print("Loading shared-grid surface tensor ...")
    diff_state = DiffusionState(args.surface_tensor)

    test_start = pd.Timestamp(args.test_start)
    test_end   = pd.Timestamp(args.test_end)
    print("Loading OptionMetrics delta-grid surface ...")
    delta_surface_lookup = load_delta_surface(
        args.data_dir, start_year=test_start.year - 1, end_year=test_end.year)
    print(f"  {len(delta_surface_lookup)} daily surfaces "
          f"({min(delta_surface_lookup):%Y-%m-%d} to {max(delta_surface_lookup):%Y-%m-%d})")

    m0_values = M0_PAPER if args.all_m0 else [args.m0]
    monthly_starts = pd.date_range(test_start, test_end, freq="MS")

    results_all = {"unhedged": [], "delta": [], "delta_vega": [], "diffusion": []}
    obs_rows = []
    n_windows = 0

    for m0 in m0_values:
        if args.all_m0:
            print(f"\n{'─'*60}\nm0 = {m0}")
        for candidate in monthly_starts:
            if not args.all_m0 and n_windows >= args.max_windows:
                break
            label = f"m0={m0} {candidate.date()}" if args.all_m0 else f"{candidate.date()}"
            print(f"  {label} ...", end=" ", flush=True)
            try:
                panel = build_instrument_panel(candidate, m0=m0, data_dir=args.data_dir)
            except Exception as e:
                print(f"SKIP (panel: {e})")
                continue

            window = run_one_window(
                panel, adapter, diff_state,
                n_scenarios=args.n_scenarios, n_val=args.n_val,
                delta_surface_lookup=delta_surface_lookup,
                cache_dir=args.scenario_cache,
                fixed_alpha=args.fixed_alpha,
                include_underlying=args.include_underlying,
                underlying_cost_bps=args.underlying_cost_bps,
            )
            if window is None:
                print("SKIP (insufficient data)")
                continue

            for method in results_all:
                results_all[method].extend(window[method])
            for i, d in enumerate(window["dates"]):
                obs_rows.append({
                    "m0": m0, "date": d,
                    "Z_unhedged": window["unhedged"][i],
                    "Z_delta": window["delta"][i],
                    "Z_delta_vega": window["delta_vega"][i],
                    "Z_diffusion": window["diffusion"][i],
                    "alpha": window["alpha"][i],
                    "V_t": window["V_t"][i],
                    "n_instruments": window["n_instruments"][i],
                    "straddle_delta": window["straddle_delta"][i],
                    "straddle_vega": window["straddle_vega"][i],
                    "hedged_delta": window["hedged_delta"][i],
                    "hedged_vega": window["hedged_vega"][i],
                })

            print(f"OK ({len(window['diffusion'])} days, "
                  f"Z_diff std={np.std(window['diffusion']):.3f})")
            n_windows += 1

    print(f"\n{'='*60}\nTotal windows: {n_windows}, observations: {len(results_all['diffusion'])}")
    if not results_all["diffusion"]:
        print("No results to report.")
        return

    print_table2(results_all)

    if args.obs_output:
        args.obs_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(obs_rows).to_csv(args.obs_output, index=False)
        print(f"\nPer-observation results saved to {args.obs_output}")
        print("Run eval_hedging.py on it for Table 2 (Covid in/out) + Figures 6/7/8/13.")


if __name__ == "__main__":
    main()
