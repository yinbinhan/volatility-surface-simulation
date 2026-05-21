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
from volgan_adapter import MONEYNESS_GRID, TAU_GRID, sample_scenarios, scenarios_to_solver_arrays
from hedging import (
    HedgePanel,
    build_instrument_panel,
    select_alpha_aic,
    solve_transaction_cost_lasso,
    DATA_DIR,
)

RISK_FREE = 0.0  # paper uses r=0


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
    surfaces_df = pd.read_csv(prepared_dir / "surfaces_transform.csv", index_col=0)
    prices_df = pd.read_csv(prepared_dir / "spx_prices.csv", parse_dates=["date"])
    dates_df = pd.read_csv(prepared_dir / "dates.csv", parse_dates=["date"])

    raw_iv = surfaces_df.values.astype(float)           # [N, 80], raw IV (not log)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_iv_rows = np.log(np.clip(raw_iv, 1e-6, None))  # [N, 80]

    closes = prices_df["close"].values.astype(float)
    log_rets = prices_df["log_return"].values.astype(float)
    dates = [pd.Timestamp(d) for d in dates_df["date"]]
    date_to_idx = {d: i for i, d in enumerate(dates)}

    return dates, log_iv_rows, closes, log_rets, date_to_idx


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


# ─── Single window backtest ───────────────────────────────────────────────────

def run_one_window(
    panel: HedgePanel,
    gen: _VolGAN.Generator,
    state_lookup,
    n_scenarios: int,
    n_val: int,
    noise_dim: int,
    device: str,
):
    """
    Run the VolGAN LASSO backtest + baselines for one hedging window.

    Returns a dict with keys "unhedged", "delta", "volgan", each a list of
    daily tracking errors Z_t = V_t - Pi_t for t = 1..T.
    Returns None if data is too sparse to run.
    """
    dates, log_iv_rows, closes, log_rets, date_to_idx = state_lookup
    trading_dates = panel.trading_dates
    target_quotes = panel.quotes[panel.quotes["role"] == "target"].copy()
    hedge_quotes = panel.quotes[panel.quotes["role"] != "target"].copy()

    target_ids = list(panel.target.sort_values(["cp_flag", "strike"])["optionid"])
    hedge_ids = list(panel.hedges.sort_values(["cp_flag", "strike"])["optionid"])

    # Contracts as DataFrames (tau will be updated at each step)
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

    # ── t=0: get initial prices ──
    t0_date = trading_dates[0]
    t0_state = get_day_state(t0_date, date_to_idx, log_iv_rows, closes, log_rets)
    if t0_state is None:
        return None
    log_iv_t0, spot_t0, r_tm1_t0, r_tm2_t0, rvol_t0 = t0_state

    t0_target_prices = get_option_prices(panel.quotes, t0_date, target_ids)
    t0_hedge_prices = get_option_prices(panel.quotes, t0_date, hedge_ids)
    if t0_target_prices is None or t0_hedge_prices is None:
        return None

    V0 = float(t0_target_prices.sum())  # straddle value at t=0
    if V0 <= 0:
        return None

    # ── AIC alpha selection at t=0 ──
    tau_t0 = float(target_contracts["tau"].iloc[0])
    tc_t0 = _set_tau(target_contracts, tau_t0)
    hc_t0 = _set_tau(hedge_contracts, tau_t0)
    phi_zero = np.zeros(len(hedge_ids))
    c_t0 = get_half_spreads(panel.quotes, t0_date, hedge_ids)

    # N training scenarios
    spots_tr, iv_tr = sample_scenarios(
        gen, log_iv_t0, spot_t0, r_tm1_t0, r_tm2_t0, rvol_t0,
        N=n_scenarios, noise_dim=noise_dim, device=device,
    )
    dV_tr, dH_tr = scenarios_to_solver_arrays(
        spots_tr, iv_tr, spot_t0, tc_t0, hc_t0,
        t0_target_prices, t0_hedge_prices, r=RISK_FREE,
    )
    # M validation scenarios
    spots_val, iv_val = sample_scenarios(
        gen, log_iv_t0, spot_t0, r_tm1_t0, r_tm2_t0, rvol_t0,
        N=n_val, noise_dim=noise_dim, device=device,
    )
    dV_val, dH_val = scenarios_to_solver_arrays(
        spots_val, iv_val, spot_t0, tc_t0, hc_t0,
        t0_target_prices, t0_hedge_prices, r=RISK_FREE,
    )

    alpha_best = select_alpha_aic(
        dV_tr, dH_tr, dV_val, dH_val,
        phi_prev=phi_zero, c_i=c_t0, g0_scale=V0,
    )

    # ── Rolling loop ──
    phi_volgan = phi_zero.copy()
    Pi_volgan = V0
    phi_delta = 0.0        # scalar (units of underlying)
    Pi_delta = V0
    Pi_unhedged = V0

    Z_volgan, Z_delta, Z_unhedged = [], [], []

    for step in range(n_days - 1):
        date_t = trading_dates[step]
        date_tp1 = trading_dates[step + 1]

        state_t = get_day_state(date_t, date_to_idx, log_iv_rows, closes, log_rets)
        if state_t is None:
            break
        log_iv_t, spot_t, r_tm1_t, r_tm2_t, rvol_t = state_t

        prices_target_t = get_option_prices(panel.quotes, date_t, target_ids)
        prices_hedge_t = get_option_prices(panel.quotes, date_t, hedge_ids)
        prices_target_tp1 = get_option_prices(panel.quotes, date_tp1, target_ids)
        prices_hedge_tp1 = get_option_prices(panel.quotes, date_tp1, hedge_ids)

        if any(x is None for x in [prices_target_t, prices_hedge_t,
                                     prices_target_tp1, prices_hedge_tp1]):
            break

        V_t = float(prices_target_t.sum())
        V_tp1 = float(prices_target_tp1.sum())
        spot_tp1 = closes[date_to_idx.get(pd.Timestamp(date_tp1), -1)]

        # Remaining tau at next step (for scenario pricing: option matures by tau_t - 1day)
        tau_t = max((expiry - date_t).days / 365, 1.0 / 365)
        tau_tp1 = max((expiry - date_tp1).days / 365, 1.0 / 365)
        c_t = get_half_spreads(panel.quotes, date_t, hedge_ids)

        # ── Unhedged ──
        Z_unhedged.append(V_tp1 - V0)

        # ── Delta baseline ──
        # Look up ATM IV from the VolGAN grid surface (tau-major flat layout)
        m_idx = int(np.argmin(np.abs(MONEYNESS_GRID - 1.0)))
        t_idx = int(np.argmin(np.abs(TAU_GRID - tau_t)))
        flat_idx = t_idx * 10 + m_idx
        sigma_atm = float(np.exp(log_iv_rows[date_to_idx[pd.Timestamp(date_t)]][flat_idx]))
        strike_t = float(target_contracts["strike"].iloc[0])
        delta_t = straddle_bs_delta(spot_t, strike_t, tau_t, sigma_atm)

        trade_cost_delta = 0.0  # no spread cost for underlying (liquid)
        psi_delta = Pi_delta - phi_delta * spot_t - trade_cost_delta
        Pi_delta_new = phi_delta * spot_tp1 + psi_delta * (1 + RISK_FREE / 252)
        Z_delta.append(V_tp1 - Pi_delta_new)
        Pi_delta = Pi_delta_new
        phi_delta = delta_t  # rebalance delta each day

        # ── VolGAN LASSO ──
        tc_t = _set_tau(target_contracts, tau_tp1)
        hc_t = _set_tau(hedge_contracts, tau_tp1)

        spots_next, iv_next = sample_scenarios(
            gen, log_iv_t, spot_t, r_tm1_t, r_tm2_t, rvol_t,
            N=n_scenarios, noise_dim=noise_dim, device=device,
        )
        dV_t, dH_t = scenarios_to_solver_arrays(
            spots_next, iv_next, spot_t, tc_t, hc_t,
            prices_target_t, prices_hedge_t, r=RISK_FREE,
        )

        result = solve_transaction_cost_lasso(
            dV_t, dH_t, phi_volgan, c_t, alpha=alpha_best, g0_scale=V0,
        )
        phi_new = result.phi
        trade_cost = float(np.dot(c_t, np.abs(result.trade)))
        psi = Pi_volgan - float(np.dot(phi_new, prices_hedge_t)) - trade_cost
        Pi_volgan_new = float(np.dot(phi_new, prices_hedge_tp1)) + psi * (1 + RISK_FREE / 252)
        Z_volgan.append(V_tp1 - Pi_volgan_new)

        phi_volgan = phi_new
        Pi_volgan = Pi_volgan_new

    if not Z_volgan:
        return None

    return {"unhedged": Z_unhedged, "delta": Z_delta, "volgan": Z_volgan}


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
                        help="Straddle moneyness (paper tests 0.75,0.8,0.9,1.1,1.2,1.25)")
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
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    gen = _VolGAN.Generator(
        noise_dim=ckpt["noise_dim"],
        cond_dim=ckpt["cond_dim"],
        hidden_dim=ckpt["hidden_dim"],
        output_dim=ckpt["out_dim"],
    ).to(args.device)
    gen.load_state_dict(ckpt["gen_state"])
    gen.eval()

    # ── Load preprocessed surface state ──
    print("Loading preprocessed surfaces ...")
    state_lookup = build_state_lookup(args.prepared_dir)
    dates, log_iv_rows, closes, log_rets, date_to_idx = state_lookup

    # ── Generate monthly start dates ──
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    # Monthly first-of-month candidates within test period
    monthly_starts = pd.date_range(test_start, test_end, freq="MS")

    covid_start = pd.Timestamp("2020-02-13")
    covid_end = pd.Timestamp("2020-07-21")

    # ── Run backtest ──
    results_all: dict[str, list[float]] = {"unhedged": [], "delta": [], "volgan": []}
    n_windows = 0

    for candidate in monthly_starts:
        if n_windows >= args.max_windows:
            break
        if args.exclude_covid and covid_start <= candidate <= covid_end:
            print(f"  Skipping Covid window {candidate.date()}")
            continue

        print(f"Window {n_windows + 1:2d}: {candidate.date()} ...", end=" ", flush=True)
        try:
            panel = build_instrument_panel(candidate, m0=args.m0, data_dir=args.data_dir)
        except Exception as e:
            print(f"SKIP (panel build failed: {e})")
            continue

        window_results = run_one_window(
            panel, gen, state_lookup,
            n_scenarios=args.n_scenarios,
            n_val=args.n_val,
            noise_dim=ckpt["noise_dim"],
            device=args.device,
        )
        if window_results is None:
            print("SKIP (insufficient data)")
            continue

        for method in results_all:
            results_all[method].extend(window_results[method])

        n_days_done = len(window_results["volgan"])
        print(f"OK ({n_days_done} days, Z_volgan std={np.std(window_results['volgan']):.3f})")
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
