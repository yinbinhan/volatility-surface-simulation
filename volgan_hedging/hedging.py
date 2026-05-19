#!/usr/bin/env python3
"""Instrument-panel utilities for data-driven hedging experiments.

The default panel builder preserves observed OptionMetrics quote coverage.  For
paper-comparable VolGAN hedging runs, ``run_daily_backtest`` can instead consume
smoothed daily IV and bid-ask-spread surfaces so selected instruments are valued
by interpolation/extrapolation rather than exact option-id quote availability.
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from datacleaning import interpolate_surface


DATA_DIR = Path("data/optionmetrics_spx_20000103_20230228")
HEDGE_MONEYNESS = (0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1)
RAW_OPTION_COLUMNS = [
    "date",
    "exdate",
    "optionid",
    "cp_flag",
    "strike",
    "strike_price",
    "spot",
    "moneyness",
    "mid_price",
    "half_spread",
    "bid_ask_spread",
    "delta",
    "vega",
    "impl_volatility",
    "days_to_exp",
    "ttm",
    "volume",
    "open_interest",
    "symbol",
]
UNDERLYING_COLUMNS = ["date", "close", "return"]

# Phase 3 deprecation warning flags (once-per-process discipline).
_DEPRECATED_TRAIN_VAL_SPLIT_WARNED: bool = False
_DEPRECATED_FIXED_ATM_WARNED: bool = False
_DEPRECATED_ALPHA_DEFER_WARNED: bool = False
_R_PCP_MISSING_WARNED: bool = False
# Phase 3 self-check instrumentation: gated counter for select_alpha_aic calls.
_SELF_CHECK_INSTRUMENT_ENABLED: bool = False
_SELECT_ALPHA_AIC_CALL_COUNTER: int = 0


@dataclass(frozen=True)
class HedgePanel:
    """Selected contracts and observed quotes for one hedging interval."""

    start_date: pd.Timestamp
    expiry_date: pd.Timestamp
    m0: float
    target: pd.DataFrame
    hedges: pd.DataFrame
    quotes: pd.DataFrame
    missing_quotes: pd.DataFrame
    trading_dates: pd.DatetimeIndex


def _as_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _years_between(start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    return list(range(start.year, end.year + 1))


def _read_existing_csvs(paths: Iterable[Path], **kwargs) -> pd.DataFrame:
    frames = [pd.read_csv(path, **kwargs) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No matching data files found")
    return pd.concat(frames, ignore_index=True)


def load_underlying(
    data_dir: Path = DATA_DIR,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load SPX underlying closes for the requested date window."""

    if start_date is None or end_date is None:
        years = range(2000, 2024)
    else:
        years = _years_between(_as_timestamp(start_date), _as_timestamp(end_date))
    paths = [data_dir / "underlying" / f"spx_secprd_{year}.csv.gz" for year in years]
    df = _read_existing_csvs(paths, usecols=UNDERLYING_COLUMNS, parse_dates=["date"])
    if start_date is not None:
        df = df[df["date"] >= _as_timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= _as_timestamp(end_date)]
    return df.sort_values("date").reset_index(drop=True)


def load_raw_options(
    data_dir: Path = DATA_DIR,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load raw SPX option rows for the requested date window."""

    if start_date is None or end_date is None:
        years = range(2000, 2024)
    else:
        years = _years_between(_as_timestamp(start_date), _as_timestamp(end_date))
    paths = [data_dir / "raw_options" / f"spx_options_{year}.csv.gz" for year in years]
    usecols = columns or RAW_OPTION_COLUMNS
    df = _read_existing_csvs(paths, usecols=usecols, parse_dates=["date", "exdate"])
    if start_date is not None:
        df = df[df["date"] >= _as_timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= _as_timestamp(end_date)]
    return df.sort_values(["date", "exdate", "cp_flag", "strike"]).reset_index(drop=True)


def first_trading_date_on_or_after(
    requested_date: str | pd.Timestamp, underlying: pd.DataFrame
) -> pd.Timestamp:
    """Return the first available underlying date on or after `requested_date`."""

    requested = _as_timestamp(requested_date)
    dates = underlying.loc[underlying["date"] >= requested, "date"]
    if dates.empty:
        raise ValueError(f"No trading date on or after {requested.date()}")
    return pd.Timestamp(dates.iloc[0]).normalize()


def choose_expiry(start_rows: pd.DataFrame, target_days: int = 30) -> pd.Timestamp:
    """Choose the available expiry nearest to a one-month maturity."""

    expiries = (
        start_rows[["exdate", "days_to_exp"]]
        .drop_duplicates()
        .assign(distance=lambda x: (x["days_to_exp"] - target_days).abs())
        .sort_values(["distance", "days_to_exp", "exdate"])
    )
    if expiries.empty:
        raise ValueError("No expiry candidates on start date")
    return pd.Timestamp(expiries.iloc[0]["exdate"]).normalize()


def _nearest_contract(rows: pd.DataFrame, cp_flag: str, strike_target: float) -> pd.Series:
    side = rows[rows["cp_flag"] == cp_flag].copy()
    if side.empty:
        raise ValueError(f"No {cp_flag} contracts available for requested selection")
    side["strike_distance"] = (side["strike"] - strike_target).abs()
    side = side.sort_values(["strike_distance", "strike", "optionid"])
    return side.iloc[0]


def select_target_straddle(
    start_rows: pd.DataFrame, start_spot: float, expiry: pd.Timestamp, m0: float
) -> pd.DataFrame:
    """Select nearest-strike call and put for the target long straddle."""

    expiry_rows = start_rows[start_rows["exdate"] == expiry]
    strike_target = m0 * start_spot
    paired_strikes = (
        expiry_rows.groupby("strike")["cp_flag"]
        .agg(lambda flags: {"C", "P"}.issubset(set(flags)))
        .loc[lambda has_pair: has_pair]
        .index.to_series()
    )
    if paired_strikes.empty:
        raise ValueError("No same-strike call/put pair available for target straddle")
    strike = paired_strikes.iloc[(paired_strikes - strike_target).abs().argsort().iloc[0]]
    strike_rows = expiry_rows[expiry_rows["strike"] == strike]
    call = _nearest_contract(strike_rows, "C", strike)
    put = _nearest_contract(strike_rows, "P", strike)
    target = pd.DataFrame([call, put]).copy()
    target.insert(0, "role", "target")
    target["target_moneyness"] = m0
    return target.reset_index(drop=True)


def select_hedge_candidates(
    start_rows: pd.DataFrame,
    start_spot: float,
    expiry: pd.Timestamp,
    target_optionids: set[float],
    hedge_moneyness: Iterable[float] = HEDGE_MONEYNESS,
    include_underlying: bool = True,
) -> pd.DataFrame:
    """Select paper-style candidate hedging instruments on the start date.

    Phase 4 / M10: include a single UNDERLYING_SPX hedge row by default so
    every panel carries exactly one underlying hedge candidate (paper §4).
    Set ``include_underlying=False`` for tests that assert option-only output.
    """

    expiry_rows = start_rows[start_rows["exdate"] == expiry]
    selected = []
    for m in hedge_moneyness:
        cp_flag = "P" if m < 1.0 else "C"
        contract = _nearest_contract(expiry_rows, cp_flag, m * start_spot).copy()
        contract["hedge_moneyness"] = float(m)
        selected.append(contract)

    hedges = pd.DataFrame(selected)
    hedges = hedges[~hedges["optionid"].isin(target_optionids)].copy()
    hedges.insert(0, "role", "hedge")
    hedges = hedges.drop_duplicates("optionid").reset_index(drop=True)
    if include_underlying:
        underlying_row = {
            "role": "hedge",
            "optionid": "UNDERLYING_SPX",
            "cp_flag": "U",
            "strike": float("nan"),
            "exdate": expiry,
            "hedge_moneyness": 1.0,
        }
        hedges = pd.concat([pd.DataFrame([underlying_row]), hedges], ignore_index=True, sort=False)
    return hedges.reset_index(drop=True)


def expected_trading_dates(
    underlying: pd.DataFrame, start_date: pd.Timestamp, expiry_date: pd.Timestamp
) -> pd.DatetimeIndex:
    dates = underlying.loc[
        (underlying["date"] >= start_date) & (underlying["date"] <= expiry_date), "date"
    ]
    return pd.DatetimeIndex(dates.drop_duplicates().sort_values())


def quote_coverage(
    quotes: pd.DataFrame, selected: pd.DataFrame, trading_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Summarize observed and missing quote dates for selected option IDs."""

    expected = len(trading_dates)
    rows = []
    for selected_row in selected.itertuples(index=False):
        inst_quotes = quotes[quotes["optionid"] == selected_row.optionid]
        observed_dates = pd.DatetimeIndex(inst_quotes["date"].drop_duplicates())
        missing = trading_dates.difference(observed_dates)
        rows.append(
            {
                "role": selected_row.role,
                "optionid": selected_row.optionid,
                "cp_flag": selected_row.cp_flag,
                "strike": selected_row.strike,
                "exdate": selected_row.exdate,
                "expected_dates": expected,
                "observed_dates": len(observed_dates),
                "missing_dates": len(missing),
                "first_missing_date": missing[0] if len(missing) else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def build_instrument_panel(
    start_date: str | pd.Timestamp,
    m0: float,
    data_dir: Path = DATA_DIR,
    target_days: int = 30,
) -> HedgePanel:
    """Build an observed-quote panel for one target straddle interval."""

    requested = _as_timestamp(start_date)
    underlying_lookup = load_underlying(
        data_dir=data_dir,
        start_date=requested,
        end_date=requested + pd.Timedelta(days=45),
    )
    actual_start = first_trading_date_on_or_after(requested, underlying_lookup)
    options_lookup = load_raw_options(
        data_dir=data_dir,
        start_date=actual_start,
        end_date=actual_start,
    )
    start_rows = options_lookup[options_lookup["date"] == actual_start].copy()
    if start_rows.empty:
        raise ValueError(f"No option rows on start date {actual_start.date()}")

    start_spot = float(
        underlying_lookup.loc[underlying_lookup["date"] == actual_start, "close"].iloc[0]
    )
    expiry = choose_expiry(start_rows, target_days=target_days)
    target = select_target_straddle(start_rows, start_spot, expiry, m0=m0)
    target_ids = set(target["optionid"])
    hedges = select_hedge_candidates(start_rows, start_spot, expiry, target_ids)
    selected = pd.concat([target, hedges], ignore_index=True, sort=False)

    underlying = load_underlying(data_dir=data_dir, start_date=actual_start, end_date=expiry)
    trading_dates = expected_trading_dates(underlying, actual_start, expiry)
    options = load_raw_options(data_dir=data_dir, start_date=actual_start, end_date=expiry)
    # Phase 4 / M10: UNDERLYING_SPX is in `selected` but never appears in the raw
    # option panel, so exclude it from the optionid-keyed merge to avoid mixing
    # object and float64 keys; per-date underlying quote rows are appended below.
    option_selected = selected[selected["optionid"].astype(str) != "UNDERLYING_SPX"].copy()
    quotes = options[options["optionid"].isin(option_selected["optionid"])].copy()
    quotes = quotes.merge(
        option_selected[["optionid", "role"]].drop_duplicates(),
        on="optionid",
        how="left",
        validate="many_to_one",
    )
    # Phase 4 / M10: inline UNDERLYING_SPX per-date quote rows so the panel's
    # hedge book always exposes the spot leg without an external add_underlying step.
    underlying_quote_rows = []
    underlying_by_date = underlying.set_index("date")["close"].to_dict()
    for trading_date in trading_dates:
        if trading_date not in underlying_by_date:
            continue
        close = float(underlying_by_date[trading_date])
        days_to_exp = max((expiry - pd.Timestamp(trading_date)).days, 1)
        underlying_quote_rows.append({
            "date": pd.Timestamp(trading_date),
            "exdate": expiry,
            "optionid": "UNDERLYING_SPX",
            "role": "hedge",
            "cp_flag": "U",
            "strike": float("nan"),
            "mid_price": close,
            "half_spread": float(close) * UNDERLYING_HALF_SPREAD_OVER_S,
            "delta": 1.0,
            "vega": 0.0,
            "spot": close,
            "days_to_exp": days_to_exp,
        })
    if underlying_quote_rows:
        quotes = pd.concat([quotes, pd.DataFrame(underlying_quote_rows)], ignore_index=True, sort=False)
    quotes = quotes.sort_values(["date", "role", "cp_flag", "strike"]).reset_index(drop=True)
    missing = quote_coverage(quotes, selected, trading_dates)

    # Phase 4 / M17-broader: panel must carry exactly one UNDERLYING_SPX hedge row.
    if int((hedges["optionid"] == "UNDERLYING_SPX").sum()) != 1:
        raise ValueError("panel.hedges must contain exactly one UNDERLYING_SPX row")

    return HedgePanel(
        start_date=actual_start,
        expiry_date=expiry,
        m0=m0,
        target=target,
        hedges=hedges,
        quotes=quotes,
        missing_quotes=missing,
        trading_dates=trading_dates,
    )


def _write_outputs(panel: HedgePanel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.target.to_csv(output_dir / "target.csv", index=False)
    panel.hedges.to_csv(output_dir / "hedges.csv", index=False)
    panel.quotes.to_csv(output_dir / "quotes_observed.csv", index=False)
    panel.missing_quotes.to_csv(output_dir / "missing_quotes.csv", index=False)


def _required_columns_present(df: pd.DataFrame) -> bool:
    required = {"mid_price", "half_spread", "delta", "vega", "spot", "strike", "days_to_exp"}
    return required.issubset(df.columns)


def panel_self_check(panel: HedgePanel) -> list[str]:
    """Return invariant violations for a panel; empty means PASS."""

    violations = []
    if panel.target.shape[0] != 2:
        violations.append("target straddle must contain exactly one call and one put")
    if set(panel.target["cp_flag"]) != {"C", "P"}:
        violations.append("target straddle must contain call and put")
    if panel.target["strike"].nunique() != 1:
        violations.append("target straddle call and put must share one strike")
    if panel.hedges.empty:
        violations.append("hedge candidate set is empty")
    if set(panel.target["optionid"]).intersection(set(panel.hedges["optionid"])):
        violations.append("target option IDs appear in hedge candidates")
    if panel.quotes.empty:
        violations.append("observed quote panel is empty")
    if not _required_columns_present(panel.quotes):
        violations.append("observed quotes are missing required quote columns")
    if panel.missing_quotes.empty:
        violations.append("missing quote coverage table is empty")
    return violations


@dataclass(frozen=True)
class SolverScenarioArrays:
    """Solver-ready simulated target and hedge changes."""

    target_changes: np.ndarray
    hedge_changes: np.ndarray


@dataclass(frozen=True)
class DirectScenarioChanges:
    """Scenario format with changes already simulated by an upstream generator."""

    target_changes: np.ndarray | Iterable[float]
    hedge_changes: np.ndarray | Iterable[Iterable[float]]


@dataclass(frozen=True)
class SelectedInstrumentValueScenarios:
    """Scenario format with current and next selected-instrument values."""

    current_target_values: np.ndarray | Iterable[float]
    current_hedge_values: np.ndarray | Iterable[float]
    next_target_values: np.ndarray | Iterable[Iterable[float]]
    next_hedge_values: np.ndarray | Iterable[Iterable[float]]


@dataclass(frozen=True)
class NormalizedPriceSurfaceScenarios:
    """Scenario format with normalized option price surfaces and next spots."""

    target_contracts: pd.DataFrame
    hedge_contracts: pd.DataFrame
    current_target_values: np.ndarray | Iterable[float]
    current_hedge_values: np.ndarray | Iterable[float]
    normalized_surface: pd.DataFrame
    spot_next: object
    scenario_col: str = "scenario_id"
    cp_col: str = "cp_flag"
    strike_col: str = "strike"
    tau_col: str = "tau"
    moneyness_col: str = "moneyness"
    normalized_price_col: str = "normalized_price"


@dataclass(frozen=True)
class IVSurfaceScenarios:
    """Scenario format with IV surfaces, next spots, and BS revaluation."""

    target_contracts: pd.DataFrame
    hedge_contracts: pd.DataFrame
    current_target_values: np.ndarray | Iterable[float]
    current_hedge_values: np.ndarray | Iterable[float]
    iv_surface: pd.DataFrame
    spot_next: object
    risk_free_rate: float = 0.0
    scenario_col: str = "scenario_id"
    cp_col: str = "cp_flag"
    strike_col: str = "strike"
    tau_col: str = "tau"
    moneyness_col: str = "moneyness"
    iv_col: str = "implied_volatility"


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_contracts(contracts: pd.DataFrame, name: str, cp_col: str, strike_col: str, tau_col: str) -> pd.DataFrame:
    if not isinstance(contracts, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    _require_columns(contracts, [cp_col, strike_col, tau_col], name)
    if contracts.empty:
        raise ValueError(f"{name} must contain at least one selected instrument")
    frame = contracts.copy().reset_index(drop=True)
    frame[cp_col] = frame[cp_col].astype(str).str.upper()
    if not set(frame[cp_col]).issubset({"C", "P"}):
        raise ValueError(f"{name} contains an instrument with no put/call pricing route")
    for col in [strike_col, tau_col]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if not np.all(np.isfinite(frame[strike_col])) or np.any(frame[strike_col] <= 0):
        raise ValueError(f"{name}.{strike_col} must contain positive finite strikes")
    if not np.all(np.isfinite(frame[tau_col])) or np.any(frame[tau_col] <= 0):
        raise ValueError(f"{name}.{tau_col} must contain positive finite time to maturity")
    return frame


def _validate_selected_value_inputs(current_target_values, current_hedge_values, next_target_values, next_hedge_values):
    current_target = _as_float_vector(current_target_values, "current_target_values")
    current_hedge = _as_float_vector(current_hedge_values, "current_hedge_values")
    next_target = _as_float_matrix(next_target_values, "next_target_values")
    next_hedge = _as_float_matrix(next_hedge_values, "next_hedge_values")
    if current_target.shape[0] == 0:
        raise ValueError("at least one target instrument is required")
    if current_hedge.shape[0] == 0:
        raise ValueError("at least one hedge instrument is required")
    if next_target.shape[1] != current_target.shape[0]:
        raise ValueError("next_target_values columns must match current target instruments")
    if next_hedge.shape[1] != current_hedge.shape[0]:
        raise ValueError("next_hedge_values columns must match current hedge instruments")
    if next_target.shape[0] != next_hedge.shape[0]:
        raise ValueError("next target and hedge values must have the same scenario count")
    if next_target.shape[0] == 0:
        raise ValueError("at least one scenario is required")
    return current_target, current_hedge, next_target, next_hedge


def adapt_direct_changes(scenarios: DirectScenarioChanges) -> SolverScenarioArrays:
    """Validate and pass through direct simulated changes."""

    target = _as_float_vector(scenarios.target_changes, "target_changes")
    hedge = _as_float_matrix(scenarios.hedge_changes, "hedge_changes")
    if target.shape[0] != hedge.shape[0]:
        raise ValueError("target_changes and hedge_changes must have the same scenario count")
    if target.shape[0] == 0 or hedge.shape[1] == 0:
        raise ValueError("at least one scenario and one hedge instrument are required")
    return SolverScenarioArrays(target_changes=target, hedge_changes=hedge)


def adapt_selected_instrument_values(scenarios: SelectedInstrumentValueScenarios) -> SolverScenarioArrays:
    """Convert current and next selected-instrument values into changes."""

    current_target, current_hedge, next_target, next_hedge = _validate_selected_value_inputs(
        scenarios.current_target_values,
        scenarios.current_hedge_values,
        scenarios.next_target_values,
        scenarios.next_hedge_values,
    )
    return SolverScenarioArrays(
        target_changes=next_target.sum(axis=1) - float(current_target.sum()),
        hedge_changes=next_hedge - current_hedge.reshape(1, -1),
    )


def _scenario_ids_and_spots(surface: pd.DataFrame, scenario_col: str, spot_next: object) -> tuple[list[object], np.ndarray]:
    _require_columns(surface, [scenario_col], "surface")
    scenario_ids = list(pd.unique(surface[scenario_col]))
    if not scenario_ids:
        raise ValueError("surface must contain at least one scenario")
    if spot_next is None:
        raise ValueError("spot_next is required for surface revaluation")
    if isinstance(spot_next, pd.Series):
        missing = [sid for sid in scenario_ids if sid not in spot_next.index]
        if missing:
            raise ValueError(f"spot_next is missing scenarios: {missing}")
        spots = spot_next.loc[scenario_ids].to_numpy(dtype=float)
    elif isinstance(spot_next, Mapping):
        missing = [sid for sid in scenario_ids if sid not in spot_next]
        if missing:
            raise ValueError(f"spot_next is missing scenarios: {missing}")
        spots = np.asarray([spot_next[sid] for sid in scenario_ids], dtype=float)
    else:
        spots = np.asarray(spot_next, dtype=float)
        if spots.ndim != 1 or spots.shape[0] != len(scenario_ids):
            raise ValueError("spot_next must provide exactly one spot per surface scenario")
    if not np.all(np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("spot_next must contain positive finite spots for every scenario")
    return scenario_ids, spots.copy()


def _nearest_surface_value(rows, cp_flag, moneyness, tau, value_col, cp_col, moneyness_col, tau_col, surface_name):
    side = rows[rows[cp_col].astype(str).str.upper() == cp_flag].copy()
    if side.empty:
        raise ValueError(f"{surface_name} has no {cp_flag} put/call pricing route")
    for col in [moneyness_col, tau_col, value_col]:
        side[col] = pd.to_numeric(side[col], errors="coerce")
    finite = np.isfinite(side[[moneyness_col, tau_col, value_col]].to_numpy(dtype=float)).all(axis=1)
    side = side.loc[finite]
    if side.empty:
        raise ValueError(f"{surface_name} has no finite surface values for route {cp_flag}")
    if np.any(side[tau_col] <= 0):
        raise ValueError(f"{surface_name}.{tau_col} must be positive for surface revaluation")
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("selected instrument tau must be positive for surface revaluation")
    coords = side[[moneyness_col, tau_col]].to_numpy(dtype=float)
    values = side[value_col].to_numpy(dtype=float)
    m_scale = max(float(np.ptp(coords[:, 0])), 1.0)
    tau_scale = max(float(np.ptp(coords[:, 1])), 1.0)
    distances = ((coords[:, 0] - moneyness) / m_scale) ** 2 + ((coords[:, 1] - tau) / tau_scale) ** 2
    return float(values[int(np.argmin(distances))])


def _revalue_normalized_surface(contracts, surface, scenario_ids, spots, cp_col, strike_col, tau_col, scenario_col, moneyness_col, normalized_price_col):
    values = np.empty((len(scenario_ids), len(contracts)), dtype=float)
    for k, (scenario_id, spot) in enumerate(zip(scenario_ids, spots)):
        rows = surface[surface[scenario_col] == scenario_id]
        if rows.empty:
            raise ValueError(f"surface is missing scenario {scenario_id}")
        for j, contract in contracts.iterrows():
            normalized_price = _nearest_surface_value(rows, contract[cp_col], float(contract[strike_col]) / float(spot), float(contract[tau_col]), normalized_price_col, cp_col, moneyness_col, tau_col, "normalized_surface")
            if normalized_price < 0:
                raise ValueError("normalized_surface prices must be nonnegative")
            values[k, j] = float(spot) * normalized_price
    return values


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _black_scholes_price(spot, strike, tau, sigma, cp_flag, risk_free_rate):
    if spot <= 0 or strike <= 0 or tau <= 0:
        raise ValueError("Black-Scholes revaluation requires positive spot, strike, and tau")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("iv_surface implied volatilities must be positive finite values")
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    discount = math.exp(-risk_free_rate * tau)
    if cp_flag == "C":
        return spot * _normal_cdf(d1) - strike * discount * _normal_cdf(d2)
    if cp_flag == "P":
        return strike * discount * _normal_cdf(-d2) - spot * _normal_cdf(-d1)
    raise ValueError("selected instrument has no put/call pricing route")


def _black_scholes_delta_vega(spot, strike, tau, sigma, cp_flag, risk_free_rate):
    if spot <= 0 or strike <= 0 or tau <= 0:
        raise ValueError("Black-Scholes Greeks require positive spot, strike, and tau")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Black-Scholes Greeks require positive finite volatility")
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    if cp_flag == "C":
        delta = _normal_cdf(d1)
    elif cp_flag == "P":
        delta = _normal_cdf(d1) - 1.0
    else:
        raise ValueError("selected instrument has no put/call Greek route")
    vega = spot * pdf * sqrt_tau
    return float(delta), float(vega)


def _revalue_iv_surface(contracts, surface, scenario_ids, spots, risk_free_rate, cp_col, strike_col, tau_col, scenario_col, moneyness_col, iv_col):
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    values = np.empty((len(scenario_ids), len(contracts)), dtype=float)
    for k, (scenario_id, spot) in enumerate(zip(scenario_ids, spots)):
        rows = surface[surface[scenario_col] == scenario_id]
        if rows.empty:
            raise ValueError(f"iv_surface is missing scenario {scenario_id}")
        for j, contract in contracts.iterrows():
            sigma = _nearest_surface_value(rows, contract[cp_col], float(contract[strike_col]) / float(spot), float(contract[tau_col]), iv_col, cp_col, moneyness_col, tau_col, "iv_surface")
            values[k, j] = _black_scholes_price(float(spot), float(contract[strike_col]), float(contract[tau_col]), sigma, contract[cp_col], float(risk_free_rate))
    return values


def _validate_surface_contract_counts(target_contracts, hedge_contracts, current_target_values, current_hedge_values):
    current_target = _as_float_vector(current_target_values, "current_target_values")
    current_hedge = _as_float_vector(current_hedge_values, "current_hedge_values")
    if current_target.shape[0] != len(target_contracts):
        raise ValueError("current_target_values length must match target contract rows")
    if current_hedge.shape[0] != len(hedge_contracts):
        raise ValueError("current_hedge_values length must match hedge contract rows")
    return current_target, current_hedge


def adapt_normalized_price_surface(scenarios: NormalizedPriceSurfaceScenarios) -> SolverScenarioArrays:
    """Revalue selected contracts from normalized price surfaces."""

    target_contracts = _validate_contracts(scenarios.target_contracts, "target_contracts", scenarios.cp_col, scenarios.strike_col, scenarios.tau_col)
    hedge_contracts = _validate_contracts(scenarios.hedge_contracts, "hedge_contracts", scenarios.cp_col, scenarios.strike_col, scenarios.tau_col)
    current_target, current_hedge = _validate_surface_contract_counts(target_contracts, hedge_contracts, scenarios.current_target_values, scenarios.current_hedge_values)
    _require_columns(scenarios.normalized_surface, [scenarios.scenario_col, scenarios.cp_col, scenarios.moneyness_col, scenarios.tau_col, scenarios.normalized_price_col], "normalized_surface")
    scenario_ids, spots = _scenario_ids_and_spots(scenarios.normalized_surface, scenarios.scenario_col, scenarios.spot_next)
    next_target = _revalue_normalized_surface(target_contracts, scenarios.normalized_surface, scenario_ids, spots, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.normalized_price_col)
    next_hedge = _revalue_normalized_surface(hedge_contracts, scenarios.normalized_surface, scenario_ids, spots, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.normalized_price_col)
    return adapt_selected_instrument_values(SelectedInstrumentValueScenarios(current_target, current_hedge, next_target, next_hedge))


def adapt_iv_surface(scenarios: IVSurfaceScenarios) -> SolverScenarioArrays:
    """Revalue selected contracts from IV surfaces with Black-Scholes."""

    target_contracts = _validate_contracts(scenarios.target_contracts, "target_contracts", scenarios.cp_col, scenarios.strike_col, scenarios.tau_col)
    hedge_contracts = _validate_contracts(scenarios.hedge_contracts, "hedge_contracts", scenarios.cp_col, scenarios.strike_col, scenarios.tau_col)
    current_target, current_hedge = _validate_surface_contract_counts(target_contracts, hedge_contracts, scenarios.current_target_values, scenarios.current_hedge_values)
    _require_columns(scenarios.iv_surface, [scenarios.scenario_col, scenarios.cp_col, scenarios.moneyness_col, scenarios.tau_col, scenarios.iv_col], "iv_surface")
    scenario_ids, spots = _scenario_ids_and_spots(scenarios.iv_surface, scenarios.scenario_col, scenarios.spot_next)
    next_target = _revalue_iv_surface(target_contracts, scenarios.iv_surface, scenario_ids, spots, scenarios.risk_free_rate, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.iv_col)
    next_hedge = _revalue_iv_surface(hedge_contracts, scenarios.iv_surface, scenario_ids, spots, scenarios.risk_free_rate, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.iv_col)
    return adapt_selected_instrument_values(SelectedInstrumentValueScenarios(current_target, current_hedge, next_target, next_hedge))


def adapt_scenarios_to_solver(scenarios: object) -> SolverScenarioArrays:
    """Return ``target_changes`` and ``hedge_changes`` for the lasso solver."""

    if isinstance(scenarios, DirectScenarioChanges):
        return adapt_direct_changes(scenarios)
    if isinstance(scenarios, SelectedInstrumentValueScenarios):
        return adapt_selected_instrument_values(scenarios)
    if isinstance(scenarios, NormalizedPriceSurfaceScenarios):
        return adapt_normalized_price_surface(scenarios)
    if isinstance(scenarios, IVSurfaceScenarios):
        return adapt_iv_surface(scenarios)
    raise TypeError(f"unsupported scenario adapter input type: {type(scenarios).__name__}")


def scenario_adapter_self_check() -> list[str]:
    """Run deterministic checks for the generator-agnostic scenario adapter."""

    failures = []

    def expect_failure(label: str, fn) -> None:
        try:
            fn()
        except (TypeError, ValueError):
            return
        failures.append(f"{label} did not fail loudly")

    direct_target = np.array([1.0, -2.0, 0.5])
    direct_hedge = np.array([[0.2, 1.0], [-0.1, 0.5], [0.0, -0.3]])
    direct = adapt_scenarios_to_solver(DirectScenarioChanges(direct_target, direct_hedge))
    if not np.allclose(direct.target_changes, direct_target) or not np.allclose(direct.hedge_changes, direct_hedge):
        failures.append("direct changes were not passed through unchanged")

    selected = adapt_scenarios_to_solver(SelectedInstrumentValueScenarios(np.array([10.0, 4.0]), np.array([5.0, 7.0]), np.array([[11.0, 5.0], [8.0, 6.0]]), np.array([[6.0, 8.0], [4.0, 7.5]])))
    if not np.allclose(selected.target_changes, np.array([2.0, 0.0])):
        failures.append("selected-instrument target values produced wrong changes")
    if not np.allclose(selected.hedge_changes, np.array([[1.0, 1.0], [-1.0, 0.5]])):
        failures.append("selected-instrument hedge values produced wrong changes")

    target_contracts = pd.DataFrame({"optionid": [101, 102], "cp_flag": ["C", "P"], "strike": [100.0, 100.0], "tau": [0.08, 0.08]})
    hedge_contracts = pd.DataFrame({"optionid": [201, 202], "cp_flag": ["C", "P"], "strike": [95.0, 105.0], "tau": [0.08, 0.08]})
    rows = []
    for scenario_id, shift in [(0, 0.0), (1, 0.01)]:
        for cp_flag in ["C", "P"]:
            for moneyness in [0.95, 1.0, 1.05]:
                base = 0.10 if cp_flag == "C" else 0.06
                rows.append({"scenario_id": scenario_id, "cp_flag": cp_flag, "moneyness": moneyness, "tau": 0.08, "normalized_price": base + shift + 0.2 * abs(moneyness - 1.0), "implied_volatility": 0.20 + shift})
    surface = pd.DataFrame(rows)
    normalized = adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.zeros(2), surface, np.array([100.0, 100.0])))
    if normalized.target_changes.shape != (2,) or normalized.hedge_changes.shape != (2, 2):
        failures.append("normalized surface revaluation returned wrong solver array shapes")
    if not np.all(np.isfinite(normalized.target_changes)) or not np.all(np.isfinite(normalized.hedge_changes)):
        failures.append("normalized surface revaluation returned non-finite arrays")
    if not np.allclose(normalized.hedge_changes[0], np.array([11.0, 7.0])):
        failures.append("hedge contract rows did not align with hedge matrix columns")

    iv = adapt_scenarios_to_solver(IVSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.zeros(2), surface, np.array([100.0, 101.0]), risk_free_rate=0.0))
    if iv.target_changes.shape != (2,) or iv.hedge_changes.shape != (2, 2):
        failures.append("IV surface revaluation returned wrong solver array shapes")
    if not np.all(np.isfinite(iv.target_changes)) or not np.all(np.isfinite(iv.hedge_changes)):
        failures.append("IV surface revaluation returned non-finite arrays")

    expect_failure("invalid direct shape", lambda: adapt_scenarios_to_solver(DirectScenarioChanges(np.ones(2), np.ones((3, 1)))))
    expect_failure("missing spot", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.zeros(2), surface, pd.Series({0: 100.0}))))
    bad_tau_contracts = target_contracts.copy()
    bad_tau_contracts.loc[0, "tau"] = 0.0
    expect_failure("nonpositive tau", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(bad_tau_contracts, hedge_contracts, np.zeros(2), np.zeros(2), surface, np.array([100.0, 100.0]))))
    call_only_surface = surface[surface["cp_flag"] == "C"].copy()
    expect_failure("missing put/call route", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.zeros(2), call_only_surface, np.array([100.0, 100.0]))))
    expect_failure("column/instrument mismatch", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.zeros(1), surface, np.array([100.0, 100.0]))))

    return failures



@dataclass(frozen=True)
class TransactionCostLassoResult:
    """Solution summary for the transaction-cost lasso hedge update."""

    phi: np.ndarray
    trade: np.ndarray
    alpha: float
    g0: float
    intercept: float
    objective_value: float
    fit_loss: float
    transaction_penalty: float
    objective_history: tuple[float, ...]
    converged: bool
    n_iter: int
    max_coordinate_change: float


def _soft_threshold(value: np.ndarray | float, threshold: np.ndarray | float) -> np.ndarray | float:
    """Apply the scalar/vector soft-thresholding operator."""

    threshold_arr = np.asarray(threshold, dtype=float)
    if np.any(threshold_arr < 0):
        raise ValueError("soft-threshold values must be nonnegative")
    value_arr = np.asarray(value, dtype=float)
    result = np.sign(value_arr) * np.maximum(np.abs(value_arr) - threshold_arr, 0.0)
    if np.isscalar(value):
        return float(result)
    return result


def _as_float_vector(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr.copy()


def _as_float_matrix(values: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr.copy()


def _validate_lasso_inputs(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    y = _as_float_vector(target_changes, "target_changes")
    x = _as_float_matrix(hedge_changes, "hedge_changes")
    phi_prev_arr = _as_float_vector(phi_prev, "phi_prev")
    costs = _as_float_vector(c_i, "c_i")
    alpha = float(alpha)
    g0 = float(g0)
    if x.shape[0] != y.shape[0]:
        raise ValueError("target_changes and hedge_changes must have the same scenario count")
    if x.shape[1] != phi_prev_arr.shape[0]:
        raise ValueError("phi_prev length must match the number of hedge instruments")
    if costs.shape[0] != phi_prev_arr.shape[0]:
        raise ValueError("c_i length must match the number of hedge instruments")
    if y.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("at least one scenario and one hedge instrument are required")
    if np.any(costs < 0):
        raise ValueError("c_i half-spread costs must be nonnegative")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be a finite nonnegative scalar")
    if not np.isfinite(g0):
        raise ValueError("g0 must be finite")
    return y, x, phi_prev_arr, costs, alpha, g0


def _transaction_cost_lasso_components(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi: np.ndarray | Iterable[float],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0: float,
    intercept: float = 0.0,
) -> tuple[float, float, float]:
    y, x, phi_prev_arr, costs, alpha, g0 = _validate_lasso_inputs(
        target_changes, hedge_changes, phi_prev, c_i, alpha, g0
    )
    phi_arr = _as_float_vector(phi, "phi")
    if phi_arr.shape[0] != phi_prev_arr.shape[0]:
        raise ValueError("phi length must match phi_prev length")

    # Paper Eq (20): free intercept A_t (not g0); no 1/2 factor on MSE.
    residual = y - float(intercept) - x @ phi_arr
    fit_loss = float(np.mean(residual * residual))
    transaction_penalty = float(alpha * float(g0) * np.sum(costs * np.abs(phi_arr - phi_prev_arr)))
    objective_value = fit_loss + transaction_penalty
    return objective_value, fit_loss, transaction_penalty


def transaction_cost_lasso_objective(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi: np.ndarray | Iterable[float],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0: float = 0.0,
    intercept: float = 0.0,
) -> float:
    """Return MSE plus weighted L1 transaction cost on trade increments.

    Paper Eq (20): the fitted hedge is ``intercept + hedge_changes @ phi`` with
    a free intercept ``A_t`` (not ``g0``); the transaction-cost penalty is
    ``alpha * g0 * sum_i c_i * abs(phi_i - phi_prev_i)`` and is applied to the
    trade from the previous hedge, not to absolute holdings. No 1/2 factor.
    """

    objective_value, _, _ = _transaction_cost_lasso_components(
        target_changes, hedge_changes, phi, phi_prev, c_i, alpha, g0, intercept
    )
    return objective_value


def solve_transaction_cost_lasso(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> TransactionCostLassoResult:
    """Solve the transaction-cost lasso hedge update by coordinate descent.

    The optimization variable is the new hedge ``phi``. Coordinate descent is
    run on trade increments ``phi - phi_prev`` so the L1 penalty has the exact
    transaction-cost interpretation used in the objective.
    """

    y, x, phi_prev_arr, costs, alpha, g0 = _validate_lasso_inputs(
        target_changes, hedge_changes, phi_prev, c_i, alpha, g0
    )
    max_iter = int(max_iter)
    tol = float(tol)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol < 0 or not np.isfinite(tol):
        raise ValueError("tol must be a finite nonnegative scalar")

    n_scenarios, n_hedges = x.shape
    trade = np.zeros(n_hedges, dtype=float)
    intercept = 0.0
    # M-new-C: centered_target drops g0 (free intercept A_t is updated below).
    centered_target = y - x @ phi_prev_arr
    residual = centered_target.copy()
    # M-new-A: soft-threshold lambdas = (1/2) * alpha * g0 * c_j
    # (subgradient of N^{-1}||r||^2 = (2/N) X^T r yields threshold alpha*g0*c/2
    # against rho = (1/N) X_j^T r after dropping the 0.5 fit factor).
    lambdas = 0.5 * float(alpha) * float(g0) * costs
    column_norms = np.mean(x * x, axis=0)

    objective_history = [
        transaction_cost_lasso_objective(y, x, phi_prev_arr, phi_prev_arr, costs, alpha, g0, intercept)
    ]
    converged = False
    max_coordinate_change = np.inf
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        # M2 / Paper Algorithm 2 p.23: update free intercept BEFORE coordinate sweep.
        intercept_old = intercept
        r_full = y - x @ (phi_prev_arr + trade)
        intercept = float(np.mean(r_full))
        residual = r_full - intercept
        max_coordinate_change = abs(intercept - intercept_old)
        for j in range(n_hedges):
            old_trade = trade[j]
            if column_norms[j] <= np.finfo(float).eps:
                new_trade = 0.0
            else:
                residual += x[:, j] * old_trade
                rho = float(np.dot(x[:, j], residual) / n_scenarios)
                new_trade = _soft_threshold(rho, lambdas[j]) / column_norms[j]
                residual -= x[:, j] * new_trade
            trade[j] = new_trade
            max_coordinate_change = max(max_coordinate_change, abs(new_trade - old_trade))

        phi = phi_prev_arr + trade
        objective_value = transaction_cost_lasso_objective(y, x, phi, phi_prev_arr, costs, alpha, g0, intercept)
        objective_history.append(objective_value)
        n_iter = iteration
        scale = max(1.0, float(np.max(np.abs(phi))))
        objective_change = abs(objective_history[-2] - objective_history[-1])
        if max_coordinate_change <= tol * scale or objective_change <= tol * scale:
            converged = True
            break

    phi = phi_prev_arr + trade
    objective_value, fit_loss, transaction_penalty = _transaction_cost_lasso_components(
        y, x, phi, phi_prev_arr, costs, alpha, g0, intercept
    )
    return TransactionCostLassoResult(
        phi=phi,
        trade=trade.copy(),
        alpha=alpha,
        g0=g0,
        intercept=float(intercept),
        objective_value=objective_value,
        fit_loss=fit_loss,
        transaction_penalty=transaction_penalty,
        objective_history=tuple(float(v) for v in objective_history),
        converged=converged,
        n_iter=n_iter,
        max_coordinate_change=float(max_coordinate_change),
    )


def select_alpha_aic(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    validation_target_changes: np.ndarray | Iterable[float],
    validation_hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    g0: float = 0.0,
    alpha_grid: np.ndarray | Iterable[float] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    return_details: bool = False,
):
    """Select alpha by validation AIC over independent validation scenarios."""

    # Phase 3 self-check instrumentation: count invocations when gated on.
    if _SELF_CHECK_INSTRUMENT_ENABLED:
        global _SELECT_ALPHA_AIC_CALL_COUNTER
        _SELECT_ALPHA_AIC_CALL_COUNTER += 1

    if alpha_grid is None:
        alpha_values = np.round(np.arange(0.01, 0.201, 0.01), 2)
    else:
        alpha_values = _as_float_vector(alpha_grid, "alpha_grid")
    if alpha_values.shape[0] == 0:
        raise ValueError("alpha_grid must contain at least one candidate")
    if np.any(alpha_values < 0):
        raise ValueError("alpha_grid candidates must be nonnegative")

    y_val = _as_float_vector(validation_target_changes, "validation_target_changes")
    x_val = _as_float_matrix(validation_hedge_changes, "validation_hedge_changes")
    phi_prev_arr = _as_float_vector(phi_prev, "phi_prev")
    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError("validation target and hedge changes must have the same scenario count")
    if x_val.shape[1] != phi_prev_arr.shape[0]:
        raise ValueError("validation hedge columns must match phi_prev length")
    if y_val.shape[0] == 0:
        raise ValueError("at least one validation scenario is required")

    rows = []
    best_row = None
    for alpha in alpha_values:
        result = solve_transaction_cost_lasso(
            target_changes,
            hedge_changes,
            phi_prev_arr,
            c_i,
            float(alpha),
            g0=g0,
            max_iter=max_iter,
            tol=tol,
        )
        # M-new-C: validation residual uses the fitted free intercept A_t, not g0.
        validation_residual = y_val - float(result.intercept) - x_val @ result.phi
        validation_mse = float(np.mean(validation_residual * validation_residual))
        # M3 / Paper Eq (24): parameter count = 1 + nnz(phi) (positions, not trades).
        active_positions = int(np.count_nonzero(np.abs(result.phi) > 1e-8))
        aic = y_val.shape[0] * np.log(max(validation_mse, np.finfo(float).tiny)) + 2.0 * (1 + active_positions)
        row = {
            "alpha": float(alpha),
            "aic": float(aic),
            "validation_mse": validation_mse,
            "active_positions": active_positions,
            "result": result,
        }
        rows.append(row)
        if best_row is None or (row["aic"], row["alpha"]) < (best_row["aic"], best_row["alpha"]):
            best_row = row

    if return_details:
        return best_row["alpha"], best_row["result"], rows
    return best_row["alpha"]


def solver_self_check() -> list[str]:
    """Run deterministic numerical checks for the transaction-cost lasso solver."""

    failures = []

    phi_prev = np.array([0.5, -0.25, 0.1])
    phi_true = np.array([1.25, -0.75, 0.4])
    g0 = 0.2
    intercept_true = 0.2
    # Centered design: zero column-sum so the free intercept is identified
    # (a constant column is not in the span of x_full_rank).
    x_full_rank = np.array(
        [
            [1.0, -0.5, 0.3],
            [-0.4, 1.0, -0.2],
            [0.2, -0.3, 0.9],
            [-0.8, 0.7, -0.1],
            [0.0, -0.9, -0.9],
        ]
    )
    x_full_rank = x_full_rank - x_full_rank.mean(axis=0, keepdims=True)
    # M-new-B: free intercept A_t (not g0) generates the OLS target.
    y_full_rank = intercept_true + x_full_rank @ phi_true
    ols_phi = np.linalg.lstsq(x_full_rank, y_full_rank - intercept_true, rcond=None)[0]
    ols_result = solve_transaction_cost_lasso(
        y_full_rank,
        x_full_rank,
        phi_prev,
        np.zeros(3),
        alpha=0.0,
        g0=g0,
        max_iter=2000,
        tol=1e-12,
    )
    if not np.allclose(ols_result.phi, ols_phi, atol=1e-8):
        failures.append("alpha=0 with zero costs did not match deterministic full-rank OLS")
    if abs(ols_result.intercept - intercept_true) >= 1e-7:
        failures.append(
            f"OLS fixture intercept mismatch: got {ols_result.intercept}, expected {intercept_true}"
        )

    shrink_x = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.3, 1.0, 0.4],
            [0.2, -0.4, 1.0],
            [1.2, 0.1, 0.2],
            [-0.2, 1.1, 0.3],
            [0.1, -0.1, 0.9],
        ]
    )
    shrink_y = g0 + shrink_x @ phi_true
    low_alpha = solve_transaction_cost_lasso(
        shrink_y, shrink_x, phi_prev, np.ones(3), alpha=0.01, g0=g0
    )
    high_alpha = solve_transaction_cost_lasso(
        shrink_y, shrink_x, phi_prev, np.ones(3), alpha=0.5, g0=g0
    )
    if np.linalg.norm(high_alpha.trade, ord=1) >= np.linalg.norm(low_alpha.trade, ord=1):
        failures.append("larger alpha did not shrink trade increments toward phi_prev")

    shared_factor = np.linspace(-1.0, 1.0, 9)
    correlated_x = np.column_stack([shared_factor, shared_factor + 0.01 * shared_factor**2])
    correlated_y = correlated_x[:, 0]
    cost_result = solve_transaction_cost_lasso(
        correlated_y,
        correlated_x,
        np.zeros(2),
        np.array([0.05, 2.0]),
        alpha=0.05,
        g0=0.0,
        max_iter=500,
    )
    if abs(cost_result.trade[1]) >= abs(cost_result.trade[0]):
        failures.append("higher-cost correlated instrument was not penalized more than cheaper instrument")

    history = np.asarray(low_alpha.objective_history, dtype=float)
    if not np.all(np.isfinite(history)):
        failures.append("objective history contains non-finite values")
    if np.any(np.diff(history) > 1e-10):
        failures.append("objective history increased during coordinate descent")

    # AIC fixture: phi_prev=0 plus a sparse phi_true so active_positions can
    # change across the grid, and adversarial training noise so the validation
    # AIC is U-shaped over alpha (interior optimum at alpha=0.1 on this grid).
    aic_phi_prev = np.zeros(3)
    aic_phi_true = np.array([1.25, -0.75, 0.0])
    train_x = shrink_x
    train_noise = np.array([0.1, 0.1, -0.3, 0.1, 0.1, -0.3])
    train_y = g0 + train_x @ aic_phi_true + train_noise
    validation_x = np.array(
        [
            [0.8, 0.1, 0.2],
            [0.1, 0.9, -0.2],
            [-0.2, 0.3, 0.7],
            [1.1, -0.1, 0.1],
        ]
    )
    validation_y = g0 + validation_x @ aic_phi_true
    grid = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 2.0])
    manual_rows = []
    for alpha in grid:
        result = solve_transaction_cost_lasso(
            train_y,
            train_x,
            aic_phi_prev,
            np.ones(3),
            float(alpha),
            g0=g0,
        )
        # M-new-C / M3: validation residual uses fitted A_t; param count = 1 + nnz(phi).
        validation_residual = validation_y - float(result.intercept) - validation_x @ result.phi
        validation_mse = float(np.mean(validation_residual * validation_residual))
        active_positions = int(np.count_nonzero(np.abs(result.phi) > 1e-8))
        aic = validation_y.shape[0] * np.log(max(validation_mse, np.finfo(float).tiny)) + 2.0 * (1 + active_positions)
        manual_rows.append(
            {
                "alpha": float(alpha),
                "aic": float(aic),
                "validation_mse": validation_mse,
                "active_positions": active_positions,
            }
        )
    expected_alpha = min(manual_rows, key=lambda row: (row["aic"], row["alpha"]))["alpha"]
    if np.isclose(expected_alpha, grid[0]):
        failures.append(f"AIC self-check fixture selected first grid entry; rows={manual_rows}")

    selected_alpha = select_alpha_aic(
        train_y,
        train_x,
        validation_y,
        validation_x,
        aic_phi_prev,
        np.ones(3),
        g0=g0,
        alpha_grid=grid,
    )
    if not np.isclose(selected_alpha, expected_alpha):
        failures.append(
            f"AIC selector selected {selected_alpha}, expected {expected_alpha}; rows={manual_rows}"
        )

    # Strengthening 1: interior optimum -- expected_alpha must not be a boundary.
    sorted_grid = np.sort(np.asarray(grid, dtype=float))
    if np.isclose(expected_alpha, sorted_grid[0]) or np.isclose(expected_alpha, sorted_grid[-1]):
        failures.append(
            f"AIC fixture optimum at boundary alpha={expected_alpha}; rows={manual_rows}"
        )

    # Strengthening 2: order-invariance -- shuffled grid must select the same alpha.
    shuffled_grid = np.random.default_rng(0).permutation(np.asarray(grid, dtype=float)).tolist()
    shuffled_alpha = select_alpha_aic(
        train_y,
        train_x,
        validation_y,
        validation_x,
        aic_phi_prev,
        np.ones(3),
        g0=g0,
        alpha_grid=shuffled_grid,
    )
    if not np.isclose(shuffled_alpha, selected_alpha):
        failures.append(
            f"AIC selector not order-invariant: canonical {selected_alpha}, shuffled {shuffled_alpha}"
        )

    # Strengthening 3: intercept-recovery fixture on the centered design.
    # Centered X means the free intercept A_t is identifiable; alpha=0, costs=0, g0=0.
    intercept_recovery_y = 1.7 + x_full_rank @ phi_true
    intercept_recovery = solve_transaction_cost_lasso(
        intercept_recovery_y,
        x_full_rank,
        phi_prev,
        np.zeros(3),
        alpha=0.0,
        g0=0.0,
        max_iter=2000,
        tol=1e-12,
    )
    if abs(intercept_recovery.intercept - 1.7) >= 1e-7:
        failures.append(
            f"intercept-recovery fixture: got {intercept_recovery.intercept}, expected 1.7"
        )
    if not np.allclose(intercept_recovery.phi, phi_true, atol=1e-7):
        failures.append(
            f"intercept-recovery fixture: phi {intercept_recovery.phi}, expected {phi_true}"
        )

    # Phase 3 addendum: second AIC fixture where phi_prev != 0 so nnz(phi) > nnz(trade)
    # at the AIC-optimal alpha.  We construct training data so the solver lands at
    # phi=[1,1,0] (one phi_prev coord stays fixed: phi[0]) at large alpha and at a
    # less shrunk phi at small alpha, and validation_y is engineered so the correct
    # AIC convention (1 + nnz(phi)) prefers a small alpha whereas the broken one
    # (1 + nnz(trade)) picks a different alpha that has fewer non-zero TRADES but
    # the same non-zero POSITIONS.  Paper Eq (24) counts positions, not trades.
    add_phi_prev = np.array([1.0, 1.0, 0.0])
    add_phi_true = np.array([1.1, 0.9, 0.05])
    add_train_x = np.array([
        [0.5, 0.4, -0.3],
        [-0.4, 0.6, 0.2],
        [0.2, -0.5, 0.5],
        [-0.3, -0.5, -0.4],
    ])
    add_train_x = add_train_x - add_train_x.mean(axis=0, keepdims=True)
    add_train_y = g0 + add_train_x @ add_phi_true
    add_validation_x = np.array([
        [0.6, 0.3, 0.1],
        [-0.5, 0.2, 0.4],
        [0.3, -0.4, -0.2],
        [-0.4, -0.1, -0.3],
        [0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2],
    ])
    add_validation_x = add_validation_x - add_validation_x.mean(axis=0, keepdims=True)
    # Validation phi target is a convex combination of the alpha=0.1 and alpha=0.3
    # solver outputs (lam=0.65 favours the alpha=0.1 solution).  This calibrates
    # MSE_val(0.1) < MSE_val(0.15) so the correct AIC (penalising nnz_phi=2 at both)
    # picks alpha=0.1, while the broken AIC (with nnz_trade=2 at 0.1 and 1 at 0.15)
    # gets a 2-point bonus at 0.15 and picks 0.15 instead.
    add_grid = np.array([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    add_solns = {}
    for alpha in add_grid:
        add_solns[float(alpha)] = solve_transaction_cost_lasso(
            add_train_y, add_train_x, add_phi_prev, np.ones(3), float(alpha), g0=g0
        )
    add_phi_a = add_solns[0.1].phi
    add_phi_b = add_solns[0.3].phi
    add_lam = 0.65
    add_phi_val_target = add_lam * add_phi_a + (1.0 - add_lam) * add_phi_b
    add_validation_y = g0 + add_validation_x @ add_phi_val_target

    add_correct_rows = []
    add_broken_rows = []
    for alpha in add_grid:
        r_alpha = add_solns[float(alpha)]
        res_val = add_validation_y - float(r_alpha.intercept) - add_validation_x @ r_alpha.phi
        mse_val = float(np.mean(res_val * res_val))
        N_v = add_validation_y.shape[0]
        nnz_phi_alpha = int(np.count_nonzero(np.abs(r_alpha.phi) > 1e-8))
        nnz_trade_alpha = int(np.count_nonzero(np.abs(r_alpha.trade) > 1e-8))
        aic_correct = N_v * np.log(max(mse_val, np.finfo(float).tiny)) + 2.0 * (1 + nnz_phi_alpha)
        aic_broken = N_v * np.log(max(mse_val, np.finfo(float).tiny)) + 2.0 * (1 + nnz_trade_alpha)
        add_correct_rows.append({"alpha": float(alpha), "aic": float(aic_correct), "nnz_phi": nnz_phi_alpha, "nnz_trade": nnz_trade_alpha})
        add_broken_rows.append({"alpha": float(alpha), "aic": float(aic_broken)})
    add_alpha_correct = min(add_correct_rows, key=lambda row: (row["aic"], row["alpha"]))["alpha"]
    add_alpha_broken = min(add_broken_rows, key=lambda row: (row["aic"], row["alpha"]))["alpha"]

    # The fixture is discriminating if at the BROKEN winner alpha, nnz(phi) > nnz(trade).
    # That gap is the lever that lets the broken convention pick a different alpha
    # than the correct one.  (At the CORRECT winner, the gap may be zero -- it's the
    # discrepancy between the two argmins that we exercise here.)
    add_broken_winner = next(row for row in add_correct_rows if np.isclose(row["alpha"], add_alpha_broken))
    if add_broken_winner["nnz_phi"] <= add_broken_winner["nnz_trade"]:
        failures.append(
            f"Phase 3 AIC addendum non-discriminating at broken alpha {add_alpha_broken}: "
            f"nnz_phi={add_broken_winner['nnz_phi']}, nnz_trade={add_broken_winner['nnz_trade']}"
        )
    if np.isclose(add_alpha_correct, add_alpha_broken):
        failures.append(
            f"Phase 3 AIC addendum fixture: correct and broken conventions agree on "
            f"alpha={add_alpha_correct}; fixture cannot detect the broken convention"
        )

    # Now invoke select_alpha_aic and confirm it matches the correct convention.
    add_selected_alpha, _add_selected_result, _ = select_alpha_aic(
        add_train_y,
        add_train_x,
        add_validation_y,
        add_validation_x,
        add_phi_prev,
        np.ones(3),
        g0=g0,
        alpha_grid=add_grid,
        return_details=True,
    )
    if not np.isclose(add_selected_alpha, add_alpha_correct):
        failures.append(
            f"Phase 3 AIC addendum: selector returned {add_selected_alpha}, "
            f"correct-convention says {add_alpha_correct}; rows={add_correct_rows}"
        )
    if np.isclose(add_selected_alpha, add_alpha_broken):
        failures.append(
            f"Phase 3 AIC addendum: selector matches the broken convention alpha={add_alpha_broken}"
        )

    # Strengthening 4: KKT residual check on an active fixture (low_alpha re-used).
    kkt_costs = np.ones(3)
    kkt_alpha = float(low_alpha.alpha)
    kkt_g0 = float(low_alpha.g0)
    kkt_phi_prev = phi_prev
    kkt_trade = low_alpha.trade
    kkt_x = shrink_x
    kkt_y = shrink_y
    kkt_N = kkt_x.shape[0]
    # Gradient of (1/N) ||y - intercept - X phi||^2 w.r.t. beta = phi - phi_prev:
    #   (2/N) X_j^T (X phi + intercept - y).
    kkt_grad = (2.0 / kkt_N) * kkt_x.T @ (
        kkt_x @ low_alpha.phi + float(low_alpha.intercept) - kkt_y
    )
    kkt_tol = 5e-6
    for j in range(kkt_trade.shape[0]):
        threshold = kkt_alpha * kkt_g0 * kkt_costs[j]
        if abs(kkt_trade[j]) > 1e-8:
            residual_j = kkt_grad[j] + threshold * np.sign(kkt_trade[j])
            if abs(residual_j) >= kkt_tol:
                failures.append(
                    f"KKT active coord {j} violated: |g+lambda*sign|={abs(residual_j):.3e} > {kkt_tol}"
                )
        else:
            if abs(kkt_grad[j]) > threshold + kkt_tol:
                failures.append(
                    f"KKT inactive coord {j} violated: |g|={abs(kkt_grad[j]):.3e} > {threshold} + {kkt_tol}"
                )

    return failures


@dataclass(frozen=True)
class BenchmarkHedgePositions:
    """Greek-matching benchmark hedge positions for one rebalance date.

    ``delta_vega`` is None when the period-inception ATM option's BS vega
    falls below ``_DV_MIN_HEDGE_VEGA`` (paper §4 p.10 stability caveat); the
    caller is expected to hold the previous delta-vega positions.
    """

    delta: np.ndarray
    delta_vega: np.ndarray | None


@dataclass(frozen=True)
class DailyBacktestResult:
    """One strategy result for one fully observed adjacent-date interval."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    strategy: str
    positions: np.ndarray
    trade: np.ndarray
    alpha: float | None
    target_change: float
    hedge_change: float
    target_delta: float
    target_vega: float
    hedge_delta_exposure: float
    hedge_vega_exposure: float
    delta_residual: float
    vega_residual: float
    transaction_cost: float
    realized_tracking_error_before_cost: float
    realized_tracking_error: float
    period_start: pd.Timestamp | None = None
    cash_drift: float = 0.0
    delta_z: float = 0.0
    cumulative_z_period: float = 0.0
    # Phase 4 / M9-psi: running Pi_t after this interval (paper Eq 4); ψ derived from Eq (5).
    pi_t_next: float = 0.0
    # Phase 3 / M4: date at which alpha was frozen for this one-month period.
    alpha_selection_date: pd.Timestamp | None = None
    # Phase 3 / M7: ATM hedge optionid cached at period_start (delta_vega only).
    benchmark_atm_optionid: object | None = None
    # Phase 4 / M8: gross initial position g_0 = Sum |target_mid| cached at the
    # first complete interval and broadcast to every row in the period.
    g0: float | None = None
    # DV vega-floor (paper §4 p.10): True when the delta-vega leg held its
    # previous positions on this rebalance because the inception ATM option
    # had |vega| < _DV_MIN_HEDGE_VEGA.  Always False on lasso / delta rows.
    period_dv_held: bool = False


@dataclass(frozen=True)
class BacktestSummary:
    """Daily backtest outputs plus intervals skipped for incomplete quotes."""

    results: pd.DataFrame
    skipped_intervals: pd.DataFrame
    skipped_interval_count: int


def _contract_value(contract: object, name: str):
    if hasattr(contract, name):
        return getattr(contract, name)
    return contract[name]


class SmoothedSurfaceMarket:
    """Surface-backed quote source following the paper preprocessing route."""

    def __init__(self, spx_daily: pd.DataFrame, price_surfaces: pd.DataFrame, risk_free_rate: float = 0.0):
        self.spx_daily = spx_daily.copy()
        self.price_surfaces = price_surfaces.copy()
        self.risk_free_rate = float(risk_free_rate)
        if not np.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        self.spx_daily["date"] = pd.to_datetime(self.spx_daily["date"]).dt.normalize()
        self.price_surfaces["date"] = pd.to_datetime(self.price_surfaces["date"]).dt.normalize()
        required_spx = {"date", "spx_close"}
        required_surface = {
            "date",
            "moneyness",
            "tau",
            "call_half_spread_over_s",
            "put_half_spread_over_s",
            "call_iv",
            "put_iv",
        }
        missing_spx = required_spx.difference(self.spx_daily.columns)
        missing_surface = required_surface.difference(self.price_surfaces.columns)
        if missing_spx:
            raise ValueError(f"spx_daily is missing columns: {sorted(missing_spx)}")
        if missing_surface:
            raise ValueError(f"price_surfaces is missing columns: {sorted(missing_surface)}")
        self.m_grid = np.asarray(sorted(self.price_surfaces["moneyness"].drop_duplicates().astype(float)), dtype=float)
        self.tau_grid = np.asarray(sorted(self.price_surfaces["tau"].drop_duplicates().astype(float)), dtype=float)
        self._spot_by_date = {pd.Timestamp(row.date).normalize(): float(row.spx_close) for row in self.spx_daily[["date", "spx_close"]].itertuples(index=False)}
        # Phase 6 / r_pcp wiring: build a per-date risk-free-rate lookup from the
        # PCP-implied column (paper §4.1, populated by Phase 5 M13).  Forward-fill
        # any NaN gaps in date order.  Fall back to ``r`` if present, otherwise
        # leave the map empty and BS pricing will use ``self.risk_free_rate``.
        self._rate_by_date: dict[pd.Timestamp, float] = {}
        for _rate_col in ("r_pcp", "r"):
            if _rate_col in self.spx_daily.columns:
                _ordered = self.spx_daily[["date", _rate_col]].sort_values("date")
                _rates = pd.to_numeric(_ordered[_rate_col], errors="coerce").ffill()
                self._rate_by_date = {
                    pd.Timestamp(d).normalize(): float(v)
                    for d, v in zip(_ordered["date"], _rates)
                    if pd.notna(v)
                }
                break
        self._surface_cache: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
        self.rate_lookup_misses = 0
        self.lookup_count = 0
        self.offgrid_lookup_count = 0
        self.lookup_moneyness_min = float("inf")
        self.lookup_moneyness_max = float("-inf")
        self.lookup_tau_min = float("inf")
        self.lookup_tau_max = float("-inf")
        self.offgrid_moneyness_below_count = 0
        self.offgrid_moneyness_above_count = 0
        self.offgrid_tau_below_count = 0
        self.offgrid_tau_above_count = 0

    def _rate_for(self, date: pd.Timestamp) -> float:
        """Return the PCP-implied per-date risk-free rate.

        Falls back to ``self.risk_free_rate`` (the constructor scalar) when the
        date is absent from the lookup, incrementing ``rate_lookup_misses``.
        """
        key = _as_timestamp(date)
        rate = self._rate_by_date.get(key)
        if rate is None:
            self.rate_lookup_misses += 1
            return self.risk_free_rate
        return float(rate)

    def spot(self, date: pd.Timestamp) -> float:
        date = _as_timestamp(date)
        if date not in self._spot_by_date:
            raise ValueError(f"missing_surface_spot:{date.date()}")
        spot = self._spot_by_date[date]
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError(f"nonpositive_surface_spot:{date.date()}")
        return spot

    def _surface_matrices(self, date: pd.Timestamp) -> dict[str, np.ndarray]:
        date = _as_timestamp(date)
        if date in self._surface_cache:
            return self._surface_cache[date]
        rows = self.price_surfaces[self.price_surfaces["date"] == date]
        if rows.empty:
            raise ValueError(f"missing_surface_date:{date.date()}")
        matrices: dict[str, np.ndarray] = {}
        for column in ("call_iv", "put_iv", "call_half_spread_over_s", "put_half_spread_over_s"):
            pivot = rows.pivot(index="moneyness", columns="tau", values=column).reindex(index=self.m_grid, columns=self.tau_grid)
            matrix = pivot.to_numpy(dtype=float)
            if matrix.shape != (len(self.m_grid), len(self.tau_grid)) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"nonfinite_surface_matrix:{date.date()}:{column}")
            matrices[column] = matrix
        self._surface_cache[date] = matrices
        return matrices

    def interpolate_value(self, date: pd.Timestamp, column: str, moneyness: float, tau: float) -> float:
        moneyness = float(moneyness)
        tau = float(tau)
        self.lookup_count += 1
        self.lookup_moneyness_min = min(self.lookup_moneyness_min, moneyness)
        self.lookup_moneyness_max = max(self.lookup_moneyness_max, moneyness)
        self.lookup_tau_min = min(self.lookup_tau_min, tau)
        self.lookup_tau_max = max(self.lookup_tau_max, tau)
        offgrid = False
        if moneyness < self.m_grid[0]:
            self.offgrid_moneyness_below_count += 1
            offgrid = True
        if moneyness > self.m_grid[-1]:
            self.offgrid_moneyness_above_count += 1
            offgrid = True
        if tau < self.tau_grid[0]:
            self.offgrid_tau_below_count += 1
            offgrid = True
        if tau > self.tau_grid[-1]:
            self.offgrid_tau_above_count += 1
            offgrid = True
        if offgrid:
            self.offgrid_lookup_count += 1
        matrix = self._surface_matrices(date)[column]
        value = interpolate_surface(
            matrix,
            self.m_grid,
            self.tau_grid,
            np.array([moneyness], dtype=float),
            np.array([tau], dtype=float),
        )
        out = float(np.asarray(value).reshape(-1)[0])
        if not np.isfinite(out):
            raise ValueError(f"nonfinite_interpolated_surface:{_as_timestamp(date).date()}:{column}")
        return out

    def diagnostics(self) -> dict[str, object]:
        return {
            "lookup_count": int(self.lookup_count),
            "offgrid_lookup_count": int(self.offgrid_lookup_count),
            "offgrid_lookup_frac": float(self.offgrid_lookup_count / self.lookup_count) if self.lookup_count else 0.0,
            "offgrid_moneyness_below_count": int(self.offgrid_moneyness_below_count),
            "offgrid_moneyness_above_count": int(self.offgrid_moneyness_above_count),
            "offgrid_tau_below_count": int(self.offgrid_tau_below_count),
            "offgrid_tau_above_count": int(self.offgrid_tau_above_count),
            "lookup_moneyness_range": [
                None if self.lookup_count == 0 else float(self.lookup_moneyness_min),
                None if self.lookup_count == 0 else float(self.lookup_moneyness_max),
            ],
            "lookup_tau_range": [
                None if self.lookup_count == 0 else float(self.lookup_tau_min),
                None if self.lookup_count == 0 else float(self.lookup_tau_max),
            ],
            "grid_moneyness_range": [float(self.m_grid[0]), float(self.m_grid[-1])],
            "grid_tau_range": [float(self.tau_grid[0]), float(self.tau_grid[-1])],
        }


    def quote_contract(self, date: pd.Timestamp, contract: object) -> dict[str, object]:
        date = _as_timestamp(date)
        cp_flag = str(_contract_value(contract, "cp_flag")).upper()
        optionid = _contract_value(contract, "optionid")
        exdate = pd.Timestamp(_contract_value(contract, "exdate")).normalize()
        spot = self.spot(date)
        hedge_moneyness = _contract_value(contract, "hedge_moneyness") if hasattr(contract, "hedge_moneyness") else None
        if cp_flag == "U":
            return {
                "date": date,
                "exdate": exdate,
                "optionid": optionid,
                "cp_flag": cp_flag,
                "strike": np.nan,
                "hedge_moneyness": hedge_moneyness,
                "mid_price": spot,
                "half_spread": float(spot) * UNDERLYING_HALF_SPREAD_OVER_S,
                "delta": 1.0,
                "vega": 0.0,
                "spot": spot,
                "days_to_exp": max((exdate - date).days, 1),
                "ttm": max((exdate - date).days / 365.0, 1.0 / 365.0),
                "quote_source": "surface",
            }
        strike = float(_contract_value(contract, "strike"))
        if not np.isfinite(strike) or strike <= 0:
            raise ValueError(f"nonpositive_contract_strike:{optionid}")
        days_to_exp = max((exdate - date).days, 1)
        tau = max(days_to_exp / 365.0, 1.0 / 365.0)
        moneyness = strike / spot
        side = "call" if cp_flag == "C" else "put" if cp_flag == "P" else None
        if side is None:
            raise ValueError(f"unsupported_contract_type:{optionid}:{cp_flag}")
        sigma = self.interpolate_value(date, f"{side}_iv", moneyness, tau)
        spread_over_s = self.interpolate_value(date, f"{side}_half_spread_over_s", moneyness, tau)
        if sigma <= 0:
            raise ValueError(f"nonpositive_interpolated_iv:{date.date()}:{optionid}")
        if spread_over_s < 0:
            raise ValueError(f"negative_interpolated_half_spread:{date.date()}:{optionid}")
        half_spread = float(spread_over_s) * spot
        # Phase 6 / r_pcp wiring: use PCP-implied per-date r so target straddle and
        # hedge BS pricing match paper §4.1 (was r=0 default; bug).
        rate = self._rate_for(date)
        mid_price = _black_scholes_price(spot, strike, tau, sigma, cp_flag, rate)
        delta, vega = _black_scholes_delta_vega(spot, strike, tau, sigma, cp_flag, rate)
        return {
            "date": date,
            "exdate": exdate,
            "optionid": optionid,
            "cp_flag": cp_flag,
            "strike": strike,
            "hedge_moneyness": hedge_moneyness,
            "moneyness": moneyness,
            "mid_price": mid_price,
            "half_spread": half_spread,
            "bid_ask_spread": 2.0 * half_spread,
            "delta": delta,
            "vega": vega,
            "impl_volatility": sigma,
            "spot": spot,
            "days_to_exp": days_to_exp,
            "ttm": tau,
            "quote_source": "surface",
        }


def load_smoothed_surface_market(processed_dir: Path, risk_free_rate: float = 0.0) -> SmoothedSurfaceMarket:
    """Load smoothed daily surfaces used for paper-comparable hedging valuation."""

    processed_dir = Path(processed_dir)
    spx_daily = pd.read_csv(processed_dir / "spx_daily.csv.gz")
    price_surfaces = pd.read_csv(processed_dir / "price_surfaces.csv.gz")
    return SmoothedSurfaceMarket(spx_daily, price_surfaces, risk_free_rate=risk_free_rate)


def _surface_quotes_for_date(
    market: SmoothedSurfaceMarket,
    date: pd.Timestamp,
    instruments: pd.DataFrame,
    role: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows = []
    missing = []
    for contract in instruments.itertuples(index=False):
        try:
            row = market.quote_contract(date, contract)
            row["role"] = role
            rows.append(row)
        except Exception as exc:
            missing.append({
                "date": _as_timestamp(date),
                "role": role,
                "optionid": getattr(contract, "optionid", None),
                "reason": "surface_valuation_error",
                "column": None,
                "error": repr(exc),
            })
    return pd.DataFrame(rows), missing


def _surface_interval_quotes(
    panel: HedgePanel,
    market: SmoothedSurfaceMarket,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_target, missing_current_target = _surface_quotes_for_date(market, start_date, panel.target, "target")
    current_hedges, missing_current_hedges = _surface_quotes_for_date(market, start_date, panel.hedges, "hedge")
    next_target, missing_next_target = _surface_quotes_for_date(market, end_date, panel.target, "target")
    next_hedges, missing_next_hedges = _surface_quotes_for_date(market, end_date, panel.hedges, "hedge")
    missing = pd.DataFrame(missing_current_target + missing_current_hedges + missing_next_target + missing_next_hedges)
    if not missing.empty:
        missing.insert(0, "start_date", _as_timestamp(start_date))
        missing.insert(1, "end_date", _as_timestamp(end_date))
    return current_target, current_hedges, next_target, next_hedges, missing


def _ordered_quotes_for_date(
    panel: HedgePanel,
    date: pd.Timestamp,
    instruments: pd.DataFrame,
    role: str,
    required_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    date = _as_timestamp(date)
    ids = list(instruments['optionid'])
    rows = panel.quotes[
        (panel.quotes['date'] == date) & panel.quotes['optionid'].isin(ids)
    ].copy()
    if rows.empty:
        rows = pd.DataFrame(index=pd.Index([], name='optionid'))
    else:
        rows = rows.drop_duplicates('optionid', keep='first').set_index('optionid')
    rows = rows.reindex(ids)

    missing = []
    for optionid, row in rows.iterrows():
        if row.isna().all():
            missing.append({'date': date, 'role': role, 'optionid': optionid, 'reason': 'missing_quote', 'column': None})
            continue
        for column in required_columns:
            value = row.get(column, np.nan)
            if pd.isna(value):
                missing.append({'date': date, 'role': role, 'optionid': optionid, 'reason': 'missing_value', 'column': column})
    return rows.reset_index(), missing


def _complete_interval_quotes(
    panel: HedgePanel, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_required = ['mid_price', 'half_spread', 'delta', 'vega']
    next_required = ['mid_price']
    current_target, missing_current_target = _ordered_quotes_for_date(panel, start_date, panel.target, 'target', current_required)
    current_hedges, missing_current_hedges = _ordered_quotes_for_date(panel, start_date, panel.hedges, 'hedge', current_required)
    next_target, missing_next_target = _ordered_quotes_for_date(panel, end_date, panel.target, 'target', next_required)
    next_hedges, missing_next_hedges = _ordered_quotes_for_date(panel, end_date, panel.hedges, 'hedge', next_required)
    missing = pd.DataFrame(missing_current_target + missing_current_hedges + missing_next_target + missing_next_hedges)
    if not missing.empty:
        missing.insert(0, 'start_date', _as_timestamp(start_date))
        missing.insert(1, 'end_date', _as_timestamp(end_date))
    return current_target, current_hedges, next_target, next_hedges, missing


def _least_norm_exposure_match(exposures: np.ndarray, target_exposure: np.ndarray) -> np.ndarray:
    matrix = np.asarray(exposures, dtype=float)
    target = np.asarray(target_exposure, dtype=float)
    if matrix.ndim != 2 or target.ndim != 1:
        raise ValueError('benchmark exposure solve requires two-dimensional exposures and one-dimensional target')
    if matrix.shape[0] != target.shape[0]:
        raise ValueError('benchmark exposure rows must match target exposure length')
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError('benchmark exposures must be finite')
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def _underlying_hedge_index(current_hedges: pd.DataFrame) -> int:
    optionids = current_hedges.get('optionid', pd.Series(index=current_hedges.index, dtype=object)).astype(str)
    cp_flags = current_hedges.get('cp_flag', pd.Series(index=current_hedges.index, dtype=object)).astype(str).str.upper()
    matches = np.flatnonzero((optionids == 'UNDERLYING_SPX').to_numpy() | (cp_flags == 'U').to_numpy())
    if matches.size != 1:
        raise ValueError(f'expected exactly one underlying hedge row, found {matches.size}')
    return int(matches[0])


def _atm_option_hedge_index(current_hedges: pd.DataFrame, underlying_index: int) -> int:
    cp_flags = current_hedges.get('cp_flag', pd.Series(index=current_hedges.index, dtype=object)).astype(str).str.upper()
    option_mask = cp_flags.isin(['C', 'P']).to_numpy()
    option_mask[underlying_index] = False
    if not option_mask.any():
        raise ValueError('delta-vega benchmark requires at least one option hedge')
    if 'moneyness' in current_hedges.columns and current_hedges['moneyness'].notna().any():
        reference = pd.to_numeric(current_hedges['moneyness'], errors='coerce').to_numpy(dtype=float)
    elif 'hedge_moneyness' in current_hedges.columns:
        reference = pd.to_numeric(current_hedges['hedge_moneyness'], errors='coerce').to_numpy(dtype=float)
    else:
        raise ValueError('delta-vega benchmark requires current moneyness or hedge_moneyness to choose ATM option')
    distances = np.where(option_mask & np.isfinite(reference), np.abs(reference - 1.0), np.inf)
    idx = int(np.argmin(distances))
    if not np.isfinite(distances[idx]):
        raise ValueError('could not identify a finite ATM option hedge')
    return idx


# Cont–Vuletić 2025 §4 p.10 ("low vega may result in unstable hedge ratios"):
# floor below which we refuse to rebalance the ATM option leg and hold the
# previous delta-vega positions instead.  Empirically, BS vega ~5e-11 at the
# K=S_0 inception strike in the final days of a 23-td period drives
# phi^1 = kappa^V / kappa^H to ~3e10 and transaction cost to ~2.5e10.
_DV_MIN_HEDGE_VEGA: float = 1.0
_DV_HELD_WARNED: bool = False
UNDERLYING_HALF_SPREAD_OVER_S = 2.5e-4  # 5 bps full bid-ask of spot (paper §4 silent on underlying spread; reasonable proxy)


def benchmark_hedge_positions(
    current_target: pd.DataFrame,
    current_hedges: pd.DataFrame,
    fixed_atm_optionid: object | None = None,
) -> BenchmarkHedgePositions:
    """Return paper benchmark positions: underlying-only Delta and underlying+ATM Delta-vega.

    Phase 3 / M7 (paper \u00a74 p.10 \"K = S_0\"): when ``fixed_atm_optionid`` is
    provided, the delta-vega benchmark hedges with the option whose ``optionid``
    matches that value (the period-inception ATM contract).  When ``None``, the
    legacy per-interval moneyness-nearest pick is used and a once-per-process
    deprecation warning is emitted.
    """

    target_delta = float(pd.to_numeric(current_target['delta'], errors='coerce').sum())
    target_vega = float(pd.to_numeric(current_target['vega'], errors='coerce').sum())
    hedge_delta = pd.to_numeric(current_hedges['delta'], errors='coerce').to_numpy(dtype=float)
    hedge_vega = pd.to_numeric(current_hedges['vega'], errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(hedge_delta)) or not np.all(np.isfinite(hedge_vega)):
        raise ValueError('benchmark hedge Greeks must be finite')

    underlying_idx = _underlying_hedge_index(current_hedges)
    delta_positions = np.zeros(len(current_hedges), dtype=float)
    delta_positions[underlying_idx] = target_delta / hedge_delta[underlying_idx]

    if fixed_atm_optionid is None:
        global _DEPRECATED_FIXED_ATM_WARNED
        if not _DEPRECATED_FIXED_ATM_WARNED:
            warnings.warn(
                "paper \u00a74 fixes ATM hedge K=S_0 at period inception; "
                "fixed_atm_optionid=None is deprecated",
                RuntimeWarning,
                stacklevel=2,
            )
            _DEPRECATED_FIXED_ATM_WARNED = True
        atm_idx = _atm_option_hedge_index(current_hedges, underlying_idx)
    else:
        matches = np.flatnonzero((current_hedges['optionid'].to_numpy() == fixed_atm_optionid))
        if matches.size == 0:
            raise ValueError(
                f"benchmark delta-vega lost its inception ATM option {fixed_atm_optionid}"
            )
        atm_idx = int(matches[0])
    atm_vega = hedge_vega[atm_idx]
    if not np.isfinite(atm_vega) or abs(atm_vega) < _DV_MIN_HEDGE_VEGA:
        # Cont–Vuletić 2025 §4 p.10: "low vega may result in unstable hedge
        # ratios phi^1_t".  Signal to run_daily_backtest that the delta-vega
        # leg must hold its previous positions for this rebalance.
        return BenchmarkHedgePositions(delta=delta_positions, delta_vega=None)
    delta_vega_positions = np.zeros(len(current_hedges), dtype=float)
    delta_vega_positions[atm_idx] = target_vega / atm_vega
    delta_vega_positions[underlying_idx] = (target_delta - hedge_delta[atm_idx] * delta_vega_positions[atm_idx]) / hedge_delta[underlying_idx]
    return BenchmarkHedgePositions(delta=delta_positions, delta_vega=delta_vega_positions)


def _scenario_source_output(
    scenario_source: object,
    panel: HedgePanel,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_target: pd.DataFrame,
    current_hedges: pd.DataFrame,
) -> object:
    if hasattr(scenario_source, 'scenarios_for_interval'):
        return scenario_source.scenarios_for_interval(panel, start_date, end_date, current_target, current_hedges)
    if callable(scenario_source):
        return scenario_source(panel, start_date, end_date, current_target, current_hedges)
    raise TypeError('scenario_source must be callable or provide scenarios_for_interval(...)')


def _adapt_train_validation_scenarios(scenario_output: object) -> tuple[SolverScenarioArrays, SolverScenarioArrays]:
    if isinstance(scenario_output, Mapping):
        train_key = 'train' if 'train' in scenario_output else 'training' if 'training' in scenario_output else None
        validation_key = 'validation' if 'validation' in scenario_output else 'val' if 'val' in scenario_output else None
        if train_key is not None and validation_key is not None:
            return (adapt_scenarios_to_solver(scenario_output[train_key]), adapt_scenarios_to_solver(scenario_output[validation_key]))
        if 'scenarios' in scenario_output:
            scenario_output = scenario_output['scenarios']

    # Phase 3 / paper §4.2: emit a once-per-process deprecation when we fall back
    # to a deterministic 70/30 split of a single batch instead of consuming an
    # explicit M-sample validation batch from the scenario source.
    global _DEPRECATED_TRAIN_VAL_SPLIT_WARNED
    if not _DEPRECATED_TRAIN_VAL_SPLIT_WARNED:
        warnings.warn(
            "paper \u00a74.2 requires an independent M-sample validation batch (N=1000, M=100); "
            "deterministic 70/30 split of one batch is deprecated and will be removed",
            RuntimeWarning,
            stacklevel=2,
        )
        _DEPRECATED_TRAIN_VAL_SPLIT_WARNED = True

    scenarios = adapt_scenarios_to_solver(scenario_output)
    n_scenarios = scenarios.target_changes.shape[0]
    if n_scenarios < 2:
        raise ValueError('at least two scenarios are required for deterministic AIC train/validation split')
    split = max(1, int(math.floor(0.7 * n_scenarios)))
    split = min(split, n_scenarios - 1)
    train = SolverScenarioArrays(target_changes=scenarios.target_changes[:split], hedge_changes=scenarios.hedge_changes[:split])
    validation = SolverScenarioArrays(target_changes=scenarios.target_changes[split:], hedge_changes=scenarios.hedge_changes[split:])
    return train, validation


def _result_row(result: DailyBacktestResult) -> dict[str, object]:
    return {
        'start_date': result.start_date,
        'end_date': result.end_date,
        'strategy': result.strategy,
        'positions': result.positions.copy(),
        'trade': result.trade.copy(),
        'alpha': result.alpha,
        'target_change': result.target_change,
        'hedge_change': result.hedge_change,
        'target_delta': result.target_delta,
        'target_vega': result.target_vega,
        'hedge_delta_exposure': result.hedge_delta_exposure,
        'hedge_vega_exposure': result.hedge_vega_exposure,
        'delta_residual': result.delta_residual,
        'vega_residual': result.vega_residual,
        'transaction_cost': result.transaction_cost,
        'realized_tracking_error_before_cost': result.realized_tracking_error_before_cost,
        'realized_tracking_error': result.realized_tracking_error,
        'abs_realized_tracking_error': abs(result.realized_tracking_error),
        'period_start': result.period_start,
        'cash_drift': result.cash_drift,
        'delta_z': result.delta_z,
        'cumulative_z_period': result.cumulative_z_period,
        'alpha_selection_date': result.alpha_selection_date,
        'benchmark_atm_optionid': result.benchmark_atm_optionid,
        'g0': result.g0,
        'period_dv_held': result.period_dv_held,
        'pi_t_next': float(getattr(result, 'pi_t_next', 0.0)),
    }


def _daily_result(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: str,
    positions: np.ndarray,
    previous_positions: np.ndarray,
    alpha: float | None,
    target_change: float,
    hedge_changes: np.ndarray,
    target_delta: float,
    target_vega: float,
    hedge_delta: np.ndarray,
    hedge_vega: np.ndarray,
    half_spreads: np.ndarray,
    current_hedge_mid: np.ndarray,
    rate: float,
    dt: float,
    period_start: pd.Timestamp | None,
    previous_cumulative_z: float,
    pi_t: float,
    alpha_selection_date: pd.Timestamp | None = None,
    benchmark_atm_optionid: object | None = None,
    g0: float | None = None,
    period_dv_held: bool = False,
) -> DailyBacktestResult:
    trade = positions - previous_positions
    hedge_change = float(hedge_changes @ positions)
    hedge_delta_exposure = float(hedge_delta @ positions)
    hedge_vega_exposure = float(hedge_vega @ positions)
    transaction_cost = float(np.sum(half_spreads * np.abs(trade)))
    before_cost = float(target_change - hedge_change)
    # M9-psi: cash drift on Z = V − Π under paper Cont–Vuletić (2025) Eq (4)–(6).
    # Eq (4): ΔΠ_t = ψ_t r_t Δt + Σ φ_t ΔH_t,  with Π_0 = V_0.
    # Eq (5): ψ_t = Π_t − Σ φ_t H_t − Σ c_t |Δφ_t|  (self-financing).
    # Eq (6): Z_t = V_t − Π_t,  Z_0 = 0.
    # ⇒  ΔZ = ΔV − Σ φ_t ΔH − ψ_t r_t Δt = (target_change − hedge_change) − ψ_t r_t Δt.
    # Existing sign convention adds back transaction_cost in realized_tracking_error,
    # so cash_drift = −ψ_t r_t Δt.  Π_t is tracked path-dependently by the caller
    # and passed in as pi_t (reset to V_0 = Σ target_mid at period_start).
    hedge_value_t = float(np.sum(positions * current_hedge_mid))
    psi_t = float(pi_t - hedge_value_t - transaction_cost)
    cash_drift = float(-psi_t * rate * dt)
    realized_tracking_error = float(before_cost + transaction_cost)
    # Z increases when tracking error or cash drift accrue (sign convention Z = V − Π).
    delta_z = float(realized_tracking_error + cash_drift)
    cumulative_z_period = float(previous_cumulative_z + delta_z)
    # Π_{t+Δt} = Π_t + ψ_t r_t Δt + Σ φ_t ΔH_t.
    pi_t_next = float(pi_t + psi_t * rate * dt + hedge_change)
    return DailyBacktestResult(
        start_date=_as_timestamp(start_date),
        end_date=_as_timestamp(end_date),
        strategy=strategy,
        positions=positions.copy(),
        trade=trade.copy(),
        alpha=alpha,
        target_change=float(target_change),
        hedge_change=hedge_change,
        target_delta=float(target_delta),
        target_vega=float(target_vega),
        hedge_delta_exposure=hedge_delta_exposure,
        hedge_vega_exposure=hedge_vega_exposure,
        delta_residual=float(target_delta - hedge_delta_exposure),
        vega_residual=float(target_vega - hedge_vega_exposure),
        transaction_cost=transaction_cost,
        realized_tracking_error_before_cost=before_cost,
        realized_tracking_error=realized_tracking_error,
        period_start=period_start,
        cash_drift=cash_drift,
        delta_z=delta_z,
        cumulative_z_period=cumulative_z_period,
        pi_t_next=pi_t_next,
        alpha_selection_date=alpha_selection_date,
        benchmark_atm_optionid=benchmark_atm_optionid,
        g0=g0,
        period_dv_held=period_dv_held,
    )


def run_daily_backtest(
    panel: HedgePanel,
    scenario_source: object,
    alpha_grid: np.ndarray | Iterable[float] | None = None,
    strategies: tuple[str, ...] = ('lasso', 'delta', 'delta_vega'),
    quote_source: SmoothedSurfaceMarket | None = None,
) -> BacktestSummary:
    """Run model-free daily hedging mechanics over adjacent panel dates.

    By default, realized evaluation uses observed selected-contract mids.  When
    ``quote_source`` is supplied, target and hedge instruments are revalued from
    smoothed daily IV and bid-ask-spread surfaces using paper-style linear
    interpolation/extrapolation.  The sign convention is: positions are
    replicating hedge holdings, so signed realized tracking error is target
    straddle mid-price change minus hedge-portfolio mid-price change minus
    transaction costs from the rebalance trade.
    """

    requested = tuple(str(strategy) for strategy in strategies)
    allowed = {'lasso', 'delta', 'delta_vega'}
    unknown = sorted(set(requested).difference(allowed))
    if unknown:
        raise ValueError(f'unsupported backtest strategies: {unknown}')
    # Phase 4 / M17-broader: panel must carry exactly one UNDERLYING_SPX hedge row.
    if int((panel.hedges['optionid'] == 'UNDERLYING_SPX').sum()) != 1:
        raise ValueError("panel.hedges must contain exactly one UNDERLYING_SPX row")
    if len(panel.trading_dates) < 2:
        return BacktestSummary(results=pd.DataFrame(), skipped_intervals=pd.DataFrame(), skipped_interval_count=0)

    # Phase 6 / r_pcp wiring: build a per-date risk-free-rate lookup from the
    # surface market's spx_daily, preferring the PCP-implied column.  Consumer
    # for Phase 5 M13.  If no quote_source is supplied or the column is absent,
    # rate_by_date stays None and we fall back to per-row column reads.
    rate_by_date: dict[pd.Timestamp, float] | None = None
    if quote_source is not None and hasattr(quote_source, 'spx_daily'):
        _spx = quote_source.spx_daily
        for _rate_col in ('r_pcp', 'r'):
            if _rate_col in _spx.columns:
                _rates = pd.to_numeric(_spx[_rate_col], errors='coerce')
                _dates = pd.to_datetime(_spx['date']).dt.normalize()
                rate_by_date = {
                    d: float(v)
                    for d, v in zip(_dates, _rates)
                    if pd.notna(v)
                }
                break

    n_hedges = len(panel.hedges)
    previous_positions = {strategy: np.zeros(n_hedges, dtype=float) for strategy in requested}
    # M6: per-strategy running cumulative Z within the active period; period_start fixed
    # at the first complete-interval rebalance date (one panel == one one-month period).
    cumulative_z = {strategy: 0.0 for strategy in requested}
    # Phase 4 / M9-psi: path-dependent Π_t per strategy, reset to V_0 at period_start.
    pi_state = {strategy: 0.0 for strategy in requested}
    period_start: pd.Timestamp | None = None
    # Phase 3 / M4: one-period \u03b1 freeze. \u03b1 and the fitted intercept are
    # selected exactly once at the first complete-interval (period_start_resolved);
    # subsequent intervals re-solve the lasso with the frozen \u03b1.  Paper \u00a74.2.
    frozen_alpha: float | None = None
    frozen_intercept: float = 0.0
    period_start_resolved: pd.Timestamp | None = None
    # Phase 3 / M7: ATM hedge optionid pinned at period inception (paper \u00a74 K=S_0).
    fixed_atm_optionid: object | None = None
    # Phase 4 / M8: gross initial position g_0 = Sum |target_mid| cached at the
    # first complete interval and reused for every lasso solve in the period.
    g0_cached: float | None = None
    rows = []
    skipped_frames = []

    for interval_idx, (start_date, end_date) in enumerate(zip(panel.trading_dates[:-1], panel.trading_dates[1:])):
        if quote_source is None:
            current_target, current_hedges, next_target, next_hedges, missing = _complete_interval_quotes(panel, start_date, end_date)
        else:
            current_target, current_hedges, next_target, next_hedges, missing = _surface_interval_quotes(panel, quote_source, start_date, end_date)
        if not missing.empty:
            skipped_frames.append(missing)
            continue

        current_target_mid = pd.to_numeric(current_target['mid_price'], errors='coerce').to_numpy(dtype=float)
        current_hedge_mid = pd.to_numeric(current_hedges['mid_price'], errors='coerce').to_numpy(dtype=float)
        next_target_mid = pd.to_numeric(next_target['mid_price'], errors='coerce').to_numpy(dtype=float)
        next_hedge_mid = pd.to_numeric(next_hedges['mid_price'], errors='coerce').to_numpy(dtype=float)
        half_spreads = pd.to_numeric(current_hedges['half_spread'], errors='coerce').to_numpy(dtype=float)
        target_delta = float(pd.to_numeric(current_target['delta'], errors='coerce').sum())
        target_vega = float(pd.to_numeric(current_target['vega'], errors='coerce').sum())
        hedge_delta = pd.to_numeric(current_hedges['delta'], errors='coerce').to_numpy(dtype=float)
        hedge_vega = pd.to_numeric(current_hedges['vega'], errors='coerce').to_numpy(dtype=float)
        target_change = float(np.sum(next_target_mid - current_target_mid))
        hedge_changes = next_hedge_mid - current_hedge_mid

        # M9: per-date risk-free rate. Paper §4.1 specifies the spot rate is PCP-implied
        # per date; Phase 5 M13 writes `r_pcp` into spx_daily. Prefer that map built
        # once at run() entry; fall back to a `r_pcp`/`r` column on current_hedges
        # (legacy); finally default to 0.0 with a one-shot warning.
        interval_rate = None
        if rate_by_date is not None:
            interval_rate = rate_by_date.get(_as_timestamp(start_date))
        if interval_rate is None:
            for _rate_col in ('r_pcp', 'r'):
                if _rate_col in current_hedges.columns:
                    rate_series = pd.to_numeric(current_hedges[_rate_col], errors='coerce').dropna()
                    if not rate_series.empty:
                        interval_rate = float(rate_series.iloc[0])
                        break
        if interval_rate is None:
            global _R_PCP_MISSING_WARNED
            if not _R_PCP_MISSING_WARNED:
                warnings.warn(
                    'no r_pcp / r per-date rate available; cash_drift will be 0 '
                    '(paper §4.1 expects PCP-implied per-date rate)',
                    RuntimeWarning,
                )
                _R_PCP_MISSING_WARNED = True
            interval_rate = 0.0
        else:
            interval_rate = float(interval_rate)
        interval_dt = float((_as_timestamp(end_date) - _as_timestamp(start_date)).days) / 365.0

        # M6: lock the period_start on the first complete interval; shared by all strategies.
        is_period_start_interval = period_start is None
        if is_period_start_interval:
            period_start = _as_timestamp(start_date)
            # Phase 4 / M9-psi: Π_0 = V_0 (paper Eq 4).  V_0 = Σ target_mid_initial.
            target_mid_initial = float(np.sum(current_target_mid))
            for _strategy in requested:
                pi_state[_strategy] = target_mid_initial
            # Phase 3 / M4: if t=0 had missing quotes, the first complete interval is
            # not the panel's t=0; warn that alpha selection has been deferred.
            if interval_idx > 0:
                global _DEPRECATED_ALPHA_DEFER_WARNED
                if not _DEPRECATED_ALPHA_DEFER_WARNED:
                    warnings.warn(
                        f"alpha selection deferred: t=0 interval has missing quotes; "
                        f"freezing alpha at first complete interval {pd.Timestamp(start_date).date()}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _DEPRECATED_ALPHA_DEFER_WARNED = True

        if 'lasso' in requested:
            if frozen_alpha is None:
                # Period inception: select alpha once with N train + M validation.
                scenario_output = _scenario_source_output(scenario_source, panel, start_date, end_date, current_target, current_hedges)
                train, validation = _adapt_train_validation_scenarios(scenario_output)
                if validation.target_changes.shape[0] < 1:
                    raise RuntimeError(
                        "paper \u00a74.2 eq (24) requires M independent validation samples; "
                        "pass --hedge-validation-samples >= 100"
                    )
                # Phase 4 / M8: g_0 = Sum |target_mid| at period inception.
                g0_cached = float(np.sum(np.abs(current_target_mid)))
                if not np.isfinite(g0_cached) or g0_cached <= 0.0:
                    raise ValueError(
                        f"invalid g_0 at first complete interval {start_date}: {g0_cached}"
                    )
                alpha, lasso_result, _ = select_alpha_aic(
                    train.target_changes,
                    train.hedge_changes,
                    validation.target_changes,
                    validation.hedge_changes,
                    previous_positions['lasso'],
                    half_spreads,
                    g0=g0_cached,
                    alpha_grid=alpha_grid,
                    return_details=True,
                )
                frozen_alpha = float(alpha)
                frozen_intercept = float(lasso_result.intercept)
                period_start_resolved = period_start
            else:
                # Subsequent intervals: re-solve with the frozen alpha (no AIC search).
                # Pass validation_samples=0 by ignoring any returned validation batch.
                scenario_output = _scenario_source_output(scenario_source, panel, start_date, end_date, current_target, current_hedges)
                train, _validation_ignored = _adapt_train_validation_scenarios(scenario_output)
                lasso_result = solve_transaction_cost_lasso(
                    train.target_changes,
                    train.hedge_changes,
                    previous_positions['lasso'],
                    half_spreads,
                    alpha=frozen_alpha,
                    g0=g0_cached if g0_cached is not None else 0.0,
                )
            daily = _daily_result(start_date, end_date, 'lasso', lasso_result.phi, previous_positions['lasso'], float(frozen_alpha), target_change, hedge_changes, target_delta, target_vega, hedge_delta, hedge_vega, half_spreads, current_hedge_mid, interval_rate, interval_dt, period_start, cumulative_z['lasso'], pi_state['lasso'], alpha_selection_date=period_start_resolved, g0=g0_cached)
            rows.append(_result_row(daily))
            previous_positions['lasso'] = lasso_result.phi.copy()
            cumulative_z['lasso'] = daily.cumulative_z_period
            pi_state['lasso'] = daily.pi_t_next

        if 'delta' in requested or 'delta_vega' in requested:
            # Phase 3 / M7: cache ATM optionid at period inception; reuse thereafter.
            if fixed_atm_optionid is None:
                underlying_idx = _underlying_hedge_index(current_hedges)
                atm_pick = _atm_option_hedge_index(current_hedges, underlying_idx)
                fixed_atm_optionid = current_hedges.iloc[atm_pick]['optionid']
            benchmarks = benchmark_hedge_positions(current_target, current_hedges, fixed_atm_optionid=fixed_atm_optionid)
            for strategy, positions in (('delta', benchmarks.delta), ('delta_vega', benchmarks.delta_vega)):
                if strategy not in requested:
                    continue
                row_atm = fixed_atm_optionid if strategy == 'delta_vega' else None
                dv_held = False
                if strategy == 'delta_vega' and positions is None:
                    # Paper §4 p.10 vega-floor branch: hold previous positions,
                    # zero trade, zero transaction cost.  Warn once per process.
                    global _DV_HELD_WARNED
                    if not _DV_HELD_WARNED:
                        warnings.warn(
                            "delta-vega benchmark held previous positions: ATM option "
                            f"vega below _DV_MIN_HEDGE_VEGA={_DV_MIN_HEDGE_VEGA} "
                            f"on {pd.Timestamp(start_date).date()} -> {pd.Timestamp(end_date).date()} "
                            "(Cont–Vuletić 2025 §4 p.10)",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        _DV_HELD_WARNED = True
                    positions = previous_positions[strategy]
                    dv_held = True
                daily = _daily_result(start_date, end_date, strategy, positions, previous_positions[strategy], None, target_change, hedge_changes, target_delta, target_vega, hedge_delta, hedge_vega, half_spreads, current_hedge_mid, interval_rate, interval_dt, period_start, cumulative_z[strategy], pi_state[strategy], benchmark_atm_optionid=row_atm, g0=g0_cached, period_dv_held=dv_held)
                rows.append(_result_row(daily))
                previous_positions[strategy] = positions.copy()
                cumulative_z[strategy] = daily.cumulative_z_period
                pi_state[strategy] = daily.pi_t_next

    skipped = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame()
    results = pd.DataFrame(rows)
    skipped_count = 0 if skipped.empty else int(skipped[['start_date', 'end_date']].drop_duplicates().shape[0])
    return BacktestSummary(results=results, skipped_intervals=skipped, skipped_interval_count=skipped_count)


_REPORTING_STRATEGY_ORDER = {'lasso': 0, 'delta': 1, 'delta_vega': 2}


def _strategy_sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or 'strategy' not in frame.columns:
        return frame.reset_index(drop=True)
    ordered = frame.copy()
    ordered['_strategy_order'] = ordered['strategy'].map(lambda value: _REPORTING_STRATEGY_ORDER.get(str(value), len(_REPORTING_STRATEGY_ORDER)))
    ordered['_strategy_name'] = ordered['strategy'].map(str)
    ordered = ordered.sort_values(['_strategy_order', '_strategy_name']).drop(columns=['_strategy_order', '_strategy_name'])
    return ordered.reset_index(drop=True)


def _finite_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return array


def _rmse(values: np.ndarray) -> float:
    finite = _finite_array(values)
    if finite.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(finite**2)))


def _one_dimensional_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return array.reshape(-1)


def tracking_error_summary(summary: BacktestSummary) -> pd.DataFrame:
    """Return realized tracking-error moments by strategy."""

    columns = [
        'strategy',
        'tracking_count',
        'mean_signed_tracking_error',
        'mean_abs_tracking_error',
        'tracking_rmse',
        'tracking_std',
        'before_cost_tracking_rmse',
    ]
    results = summary.results
    if results.empty or 'strategy' not in results.columns or 'realized_tracking_error' not in results.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for strategy, group in results.groupby('strategy', sort=False):
        errors = _finite_array(pd.to_numeric(group['realized_tracking_error'], errors='coerce').to_numpy(dtype=float))
        before_cost_rmse = float('nan')
        if 'realized_tracking_error_before_cost' in group.columns:
            before_cost = _finite_array(pd.to_numeric(group['realized_tracking_error_before_cost'], errors='coerce').to_numpy(dtype=float))
            if before_cost.size == errors.size:
                before_cost_rmse = _rmse(before_cost)
        rows.append({
            'strategy': strategy,
            'tracking_count': int(errors.size),
            'mean_signed_tracking_error': float(np.mean(errors)) if errors.size else float('nan'),
            'mean_abs_tracking_error': float(np.mean(np.abs(errors))) if errors.size else float('nan'),
            'tracking_rmse': _rmse(errors),
            'tracking_std': float(np.std(errors, ddof=0)) if errors.size else float('nan'),
            'before_cost_tracking_rmse': before_cost_rmse,
        })
    return _strategy_sort_frame(pd.DataFrame(rows, columns=columns))


def transaction_cost_summary(summary: BacktestSummary) -> pd.DataFrame:
    """Return transaction-cost totals and moments by strategy."""

    columns = ['strategy', 'cost_count', 'transaction_cost_total', 'transaction_cost_mean', 'transaction_cost_max']
    results = summary.results
    if results.empty or 'strategy' not in results.columns or 'transaction_cost' not in results.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for strategy, group in results.groupby('strategy', sort=False):
        costs = _finite_array(pd.to_numeric(group['transaction_cost'], errors='coerce').to_numpy(dtype=float))
        rows.append({
            'strategy': strategy,
            'cost_count': int(costs.size),
            'transaction_cost_total': float(np.sum(costs)) if costs.size else float('nan'),
            'transaction_cost_mean': float(np.mean(costs)) if costs.size else float('nan'),
            'transaction_cost_max': float(np.max(costs)) if costs.size else float('nan'),
        })
    return _strategy_sort_frame(pd.DataFrame(rows, columns=columns))


def selected_hedge_count_turnover_summary(summary: BacktestSummary, tol: float = 1e-12) -> pd.DataFrame:
    """Return selected-position counts and turnover diagnostics by strategy."""

    columns = [
        'strategy',
        'activity_count',
        'nonzero_positions',
        'max_nonzero_positions',
        'nonzero_trades',
        'max_nonzero_trades',
        'l1_turnover',
        'mean_l1_turnover',
        'gross_position',
        'max_gross_position',
    ]
    results = summary.results
    required = {'strategy', 'positions', 'trade'}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame(columns=columns)

    rows = []
    for strategy, group in results.groupby('strategy', sort=False):
        positions = [_one_dimensional_array(value) for value in group['positions']]
        trades = [_one_dimensional_array(value) for value in group['trade']]
        nonzero_positions = np.array([np.count_nonzero(np.abs(value) > tol) for value in positions], dtype=float)
        nonzero_trades = np.array([np.count_nonzero(np.abs(value) > tol) for value in trades], dtype=float)
        l1_turnovers = np.array([np.sum(np.abs(value)) for value in trades], dtype=float)
        gross_positions = np.array([np.sum(np.abs(value)) for value in positions], dtype=float)
        rows.append({
            'strategy': strategy,
            'activity_count': int(len(group)),
            'nonzero_positions': float(np.mean(nonzero_positions)) if nonzero_positions.size else float('nan'),
            'max_nonzero_positions': int(np.max(nonzero_positions)) if nonzero_positions.size else 0,
            'nonzero_trades': float(np.mean(nonzero_trades)) if nonzero_trades.size else float('nan'),
            'max_nonzero_trades': int(np.max(nonzero_trades)) if nonzero_trades.size else 0,
            'l1_turnover': float(np.sum(l1_turnovers)) if l1_turnovers.size else float('nan'),
            'mean_l1_turnover': float(np.mean(l1_turnovers)) if l1_turnovers.size else float('nan'),
            'gross_position': float(np.mean(gross_positions)) if gross_positions.size else float('nan'),
            'max_gross_position': float(np.max(gross_positions)) if gross_positions.size else float('nan'),
        })
    return _strategy_sort_frame(pd.DataFrame(rows, columns=columns))


def greek_residual_summary(summary: BacktestSummary) -> pd.DataFrame:
    """Return delta/vega residual moments by strategy."""

    columns = [
        'strategy',
        'residual_count',
        'mean_abs_delta_residual',
        'delta_residual_rmse',
        'mean_abs_vega_residual',
        'vega_residual_rmse',
    ]
    results = summary.results
    required = {'strategy', 'delta_residual', 'vega_residual'}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame(columns=columns)

    rows = []
    for strategy, group in results.groupby('strategy', sort=False):
        delta_residuals = _finite_array(pd.to_numeric(group['delta_residual'], errors='coerce').to_numpy(dtype=float))
        vega_residuals = _finite_array(pd.to_numeric(group['vega_residual'], errors='coerce').to_numpy(dtype=float))
        rows.append({
            'strategy': strategy,
            'residual_count': int(min(delta_residuals.size, vega_residuals.size)),
            'mean_abs_delta_residual': float(np.mean(np.abs(delta_residuals))) if delta_residuals.size else float('nan'),
            'delta_residual_rmse': _rmse(delta_residuals),
            'mean_abs_vega_residual': float(np.mean(np.abs(vega_residuals))) if vega_residuals.size else float('nan'),
            'vega_residual_rmse': _rmse(vega_residuals),
        })
    return _strategy_sort_frame(pd.DataFrame(rows, columns=columns))


def skip_summary(summary: BacktestSummary) -> pd.DataFrame:
    """Return skipped-interval counts by reason plus the total unique skipped intervals."""

    columns = ['reason', 'skipped_rows', 'unique_skipped_intervals', 'total_unique_skipped_intervals']
    skipped = summary.skipped_intervals
    if skipped.empty or 'reason' not in skipped.columns:
        return pd.DataFrame(columns=columns)

    interval_columns = [column for column in ('start_date', 'end_date') if column in skipped.columns]
    total_unique = int(skipped[interval_columns].drop_duplicates().shape[0]) if len(interval_columns) == 2 else int(summary.skipped_interval_count)
    rows = []
    for reason, group in skipped.groupby('reason', sort=True):
        unique_intervals = int(group[interval_columns].drop_duplicates().shape[0]) if len(interval_columns) == 2 else int(len(group))
        rows.append({
            'reason': reason,
            'skipped_rows': int(len(group)),
            'unique_skipped_intervals': unique_intervals,
            'total_unique_skipped_intervals': total_unique,
        })
    return pd.DataFrame(rows, columns=columns).sort_values('reason').reset_index(drop=True)


def strategy_comparison_table(summary: BacktestSummary) -> pd.DataFrame:
    """Combine paper-level backtest diagnostics into one deterministic table."""

    frames = [
        tracking_error_summary(summary),
        transaction_cost_summary(summary),
        selected_hedge_count_turnover_summary(summary),
        greek_residual_summary(summary),
    ]
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=['strategy', 'total_unique_skipped_intervals', 'skipped_reason_count'])

    combined = nonempty[0]
    for frame in nonempty[1:]:
        combined = combined.merge(frame, on='strategy', how='outer')

    skipped = skip_summary(summary)
    combined['total_unique_skipped_intervals'] = int(skipped['total_unique_skipped_intervals'].max()) if not skipped.empty else int(summary.skipped_interval_count)
    combined['skipped_reason_count'] = int(skipped.shape[0])
    return _strategy_sort_frame(combined)


class DeterministicBacktestScenarioSource:
    """Test-only scenario source for backtest mechanics; no generator required."""

    def scenarios_for_interval(self, panel, start_date, end_date, current_target, current_hedges):
        n_hedges = len(current_hedges)
        base = np.linspace(-1.5, 1.5, 8)
        hedge_changes = np.column_stack([base * (0.2 + 0.1 * j) + 0.03 * (j + 1) * base**2 for j in range(n_hedges)])
        weights = np.linspace(0.7, 0.2, n_hedges)
        target_changes = hedge_changes @ weights + 0.05 * np.sin(np.arange(base.shape[0]))
        return DirectScenarioChanges(target_changes=target_changes, hedge_changes=hedge_changes)


def _backtest_self_check_panel() -> HedgePanel:
    dates = pd.DatetimeIndex(['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08'])
    spots = [100.0, 102.0, 108.0, 113.0, 115.0]
    target = pd.DataFrame({'role': ['target', 'target'], 'optionid': [101, 102], 'cp_flag': ['C', 'P'], 'strike': [100.0, 100.0], 'exdate': [pd.Timestamp('2020-02-03'), pd.Timestamp('2020-02-03')]})
    hedges = pd.DataFrame({'role': ['hedge', 'hedge', 'hedge', 'hedge'], 'optionid': ['UNDERLYING_SPX', 201, 202, 203], 'cp_flag': ['U', 'P', 'C', 'C'], 'strike': [0.0, 95.0, 100.0, 105.0], 'exdate': [pd.Timestamp('2020-02-03')] * 4})
    quote_specs = {
        101: ('target', 'C', 100.0, [5.0, 5.4, 5.1, 5.6, 5.8], 0.10, [0.52, 0.55, 0.50, 0.58, 0.60], [0.20, 0.19, 0.18, 0.17, 0.16]),
        102: ('target', 'P', 100.0, [4.8, 4.5, 4.9, 4.3, 4.1], 0.10, [-0.48, -0.45, -0.50, -0.42, -0.40], [0.21, 0.20, 0.19, 0.18, 0.17]),
        'UNDERLYING_SPX': ('hedge', 'U', 0.0, spots, spots[0] * UNDERLYING_HALF_SPREAD_OVER_S, [1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]),
        201: ('hedge', 'P', 95.0, [2.1, 2.0, 2.2, 1.9, 1.8], 0.04, [-0.22, -0.20, -0.24, -0.18, -0.16], [0.12, 0.11, 0.10, 0.09, 0.08]),
        202: ('hedge', 'C', 100.0, [3.2, 3.5, 3.3, 3.7, 3.9], 0.05, [0.50, 0.53, 0.49, 0.56, 0.58], [0.18, 0.17, 0.16, 0.15, 0.14]),
        203: ('hedge', 'C', 105.0, [1.7, 1.9, 1.8, 2.0, 2.1], 0.03, [0.28, 0.30, 0.27, 0.33, 0.35], [0.13, 0.12, 0.11, 0.10, 0.09]),
    }
    rows = []
    for optionid, (role, cp_flag, strike, mids, half_spread, deltas, vegas) in quote_specs.items():
        for idx, date in enumerate(dates):
            if optionid == 203 and date == dates[-1]:
                continue
            spot = spots[idx]
            moneyness = float('nan') if cp_flag == 'U' else (strike / spot)
            rows.append({'date': date, 'exdate': pd.Timestamp('2020-02-03'), 'optionid': optionid, 'role': role, 'cp_flag': cp_flag, 'strike': strike, 'mid_price': mids[idx], 'half_spread': half_spread, 'delta': deltas[idx], 'vega': vegas[idx], 'spot': spot, 'days_to_exp': 30 - idx, 'moneyness': moneyness})
    quotes = pd.DataFrame(rows)
    missing = quote_coverage(quotes, pd.concat([target, hedges], ignore_index=True), dates)
    return HedgePanel(start_date=dates[0], expiry_date=pd.Timestamp('2020-02-03'), m0=1.0, target=target, hedges=hedges, quotes=quotes, missing_quotes=missing, trading_dates=dates)


def backtest_self_check() -> list[str]:
    """Run deterministic checks for daily backtest mechanics."""

    failures = []
    panel = _backtest_self_check_panel()
    # Phase 3: instrument select_alpha_aic with a call counter so we can assert
    # M4 freezes alpha to a single AIC search per one-month period.
    global _SELF_CHECK_INSTRUMENT_ENABLED, _SELECT_ALPHA_AIC_CALL_COUNTER
    _SELF_CHECK_INSTRUMENT_ENABLED = True
    _SELECT_ALPHA_AIC_CALL_COUNTER = 0
    try:
        summary = run_daily_backtest(panel, DeterministicBacktestScenarioSource(), alpha_grid=np.array([0.0, 0.01, 0.05]))
        observed_select_alpha_calls = int(_SELECT_ALPHA_AIC_CALL_COUNTER)
    finally:
        _SELF_CHECK_INSTRUMENT_ENABLED = False
    if summary.skipped_interval_count != 1:
        failures.append(f'expected one skipped interval, found {summary.skipped_interval_count}')
    # Phase 3: 3 complete intervals \u00d7 3 strategies = 9 result rows.
    if summary.results.shape[0] != 9:
        failures.append(f'expected nine strategy rows for three complete intervals, found {summary.results.shape[0]}')
    if set(summary.results.get('strategy', [])) != {'lasso', 'delta', 'delta_vega'}:
        failures.append('backtest did not emit all expected strategies')
    if summary.results.empty:
        return failures
    numeric_columns = ['target_change', 'hedge_change', 'target_delta', 'target_vega', 'hedge_delta_exposure', 'hedge_vega_exposure', 'delta_residual', 'vega_residual', 'transaction_cost', 'realized_tracking_error_before_cost', 'realized_tracking_error']
    for column in numeric_columns:
        if not np.all(np.isfinite(summary.results[column].to_numpy(dtype=float))):
            failures.append(f'{column} contains non-finite values')
    if np.any(summary.results['transaction_cost'].to_numpy(dtype=float) < -1e-12):
        failures.append('transaction costs must be nonnegative')
    # M6/M9: new accounting columns must be finite and structurally consistent.
    new_numeric = ['cash_drift', 'delta_z', 'cumulative_z_period']
    for column in new_numeric:
        if column not in summary.results.columns:
            failures.append(f'{column} column missing from backtest results')
        elif not np.all(np.isfinite(summary.results[column].to_numpy(dtype=float))):
            failures.append(f'{column} contains non-finite values')
    if 'period_start' not in summary.results.columns:
        failures.append('period_start column missing from backtest results')
    elif summary.results['period_start'].isna().any():
        failures.append('period_start must be non-null on complete-interval rows')
    for strategy_name, group in summary.results.groupby('strategy', sort=False):
        ordered = group.sort_values('start_date').reset_index(drop=True)
        running = 0.0
        for _, row in ordered.iterrows():
            running = float(running) + float(row['delta_z'])
            if not np.isclose(float(row['cumulative_z_period']), running):
                failures.append(f"{strategy_name} cumulative_z_period does not accumulate delta_z")
                break
    first = summary.results.iloc[0]
    expected_net = float(first['target_change']) - float(first['hedge_change']) + float(first['transaction_cost'])
    if not np.isclose(float(first['realized_tracking_error']), expected_net):
        failures.append('realized tracking error does not match documented sign convention')
    expected_delta_z = float(first['realized_tracking_error']) + float(first['cash_drift'])
    if not np.isclose(float(first['delta_z']), expected_delta_z):
        failures.append('delta_z does not match realized_tracking_error + cash_drift')
    if summary.skipped_intervals.empty or 'missing_quote' not in set(summary.skipped_intervals['reason']):
        failures.append('skipped interval table did not preserve missing quote reason')

    # Phase 3 strengthenings -----------------------------------------------------
    # (i) M4: alpha must be uniform across the period for lasso rows.
    lasso_rows = summary.results[summary.results['strategy'] == 'lasso']
    if not lasso_rows.empty:
        if lasso_rows.groupby('period_start')['alpha'].nunique().max() != 1:
            failures.append('Phase 3 M4: lasso alpha is not frozen per period_start')
        # (ii) alpha_selection_date column populated on every lasso row and matches period_start.
        if 'alpha_selection_date' not in summary.results.columns:
            failures.append('Phase 3 M4: alpha_selection_date column missing from results')
        elif lasso_rows['alpha_selection_date'].isna().any():
            failures.append('Phase 3 M4: alpha_selection_date must be non-null on lasso rows')
        elif not (lasso_rows['alpha_selection_date'] == lasso_rows['period_start']).all():
            failures.append('Phase 3 M4: alpha_selection_date must equal period_start on lasso rows')
    # (iii) Counter must show exactly one select_alpha_aic call across all intervals.
    if observed_select_alpha_calls != 1:
        failures.append(
            f'Phase 3 M4: select_alpha_aic called {observed_select_alpha_calls} times, expected 1'
        )
    # (iv) M7: benchmark_atm_optionid must be identical across all delta_vega rows.
    dv_rows = summary.results[summary.results['strategy'] == 'delta_vega']
    if not dv_rows.empty:
        if 'benchmark_atm_optionid' not in summary.results.columns:
            failures.append('Phase 3 M7: benchmark_atm_optionid column missing from results')
        elif dv_rows['benchmark_atm_optionid'].isna().any():
            failures.append('Phase 3 M7: benchmark_atm_optionid must be non-null on delta_vega rows')
        elif dv_rows['benchmark_atm_optionid'].nunique() != 1:
            failures.append(
                f"Phase 3 M7: benchmark_atm_optionid drifted across period; "
                f"unique={sorted(set(dv_rows['benchmark_atm_optionid']))}"
            )
        # delta rows should NOT carry an ATM optionid (no-op on delta).
        delta_only = summary.results[summary.results['strategy'] == 'delta']
        if not delta_only.empty and not delta_only['benchmark_atm_optionid'].isna().all():
            failures.append('Phase 3 M7: benchmark_atm_optionid must be None on delta rows')

    # Phase 4 strengthenings -----------------------------------------------------
    # (v) M8: g0 column populated, finite, strictly positive, uniform per period_start
    #         on lasso rows.
    if 'g0' not in summary.results.columns:
        failures.append('Phase 4 M8: g0 column missing from backtest results')
    else:
        lasso_g0 = summary.results[summary.results['strategy'] == 'lasso']
        if not lasso_g0.empty:
            if lasso_g0['g0'].isna().any():
                failures.append('Phase 4 M8: g0 must be non-null on lasso rows')
            g0_values = pd.to_numeric(lasso_g0['g0'], errors='coerce').to_numpy(dtype=float)
            if not np.all(np.isfinite(g0_values)):
                failures.append('Phase 4 M8: g0 must be finite on lasso rows')
            elif np.any(g0_values <= 0.0):
                failures.append('Phase 4 M8: g0 must be strictly positive on lasso rows')
            if int(lasso_g0.groupby('period_start')['g0'].nunique().max()) != 1:
                failures.append('Phase 4 M8: g0 must be uniform per period_start on lasso rows')

    # (vi) M17-broader: backtest-self-check panel must contain exactly one
    #      UNDERLYING_SPX row in panel.hedges (Phase 1 added it; Phase 4 asserts).
    if int((panel.hedges['optionid'] == 'UNDERLYING_SPX').sum()) != 1:
        failures.append(
            'Phase 4 M17-broader: _backtest_self_check_panel must contain exactly one UNDERLYING_SPX hedge row'
        )

    # (vii) M12: --hedge-debug-fast must override the paper defaults via the
    #       shared _apply_debug_fast_overrides helper in volgan_experiment.py.
    try:
        import importlib
        ve = importlib.import_module('volgan_experiment')
        import argparse as _ap
        ns = _ap.Namespace(
            hedge_samples=1000,
            hedge_validation_samples=100,
            hedge_max_periods=52,
            hedge_schedule='paper_23td',
            min_hedge_observed_frac=0.0,
            hedge_quote_source='surface',
            hedge_debug_fast=True,
        )
        ve._apply_debug_fast_overrides(ns)
        if (ns.hedge_samples, ns.hedge_validation_samples, ns.hedge_max_periods,
            ns.hedge_schedule, ns.min_hedge_observed_frac, ns.hedge_quote_source) != (
            512, 0, 6, 'calendar31', 0.85, 'observed'):
            failures.append(
                f'Phase 4 M12: _apply_debug_fast_overrides did not restore debug defaults; got {vars(ns)}'
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        failures.append(f'Phase 4 M12: _apply_debug_fast_overrides import/exec failed: {exc!r}')

    # (ix) Delta tx_cost (paper §4 p.8 Eq (20)): the delta strategy hedges with
    #      the underlying only, so a nonzero trade on the underlying leg must
    #      charge the underlying half-spread. Assert sign and magnitude.
    delta_rows_tx = summary.results[summary.results['strategy'] == 'delta']
    if delta_rows_tx.empty:
        failures.append('Delta tx_cost: no delta rows emitted by backtest')
    else:
        try:
            trades_d = np.stack([np.asarray(t, dtype=float) for t in delta_rows_tx['trade'].tolist()], axis=0)
        except Exception as exc:
            trades_d = None
            failures.append(f'Delta tx_cost: could not stack trade column ({exc!r})')
        if trades_d is not None:
            tc_d = delta_rows_tx['transaction_cost'].to_numpy(dtype=float)
            # Underlying is column 0 of the delta hedge basis (single-instrument).
            underlying_trade = trades_d[:, 0] if trades_d.ndim == 2 and trades_d.shape[1] >= 1 else trades_d.reshape(-1)
            mask_nonzero = np.abs(underlying_trade) > 1e-12
            if not mask_nonzero.any():
                failures.append('Delta tx_cost: every delta row has zero underlying trade; cannot validate spread accounting')
            else:
                if np.any(tc_d[mask_nonzero] <= 0.0):
                    failures.append('Delta tx_cost: rows with nonzero underlying trade must have transaction_cost > 0')
                # Magnitude: TC = sum_i |trade_i| * half_spread_i. With single underlying leg,
                # spot=100 and |trade|=1 we expect 100 * 2.5e-4 = 0.025. Use the fixture's spot[0].
                expected_unit_tc = 100.0 * UNDERLYING_HALF_SPREAD_OVER_S
                # Check that observed TC tracks |trade| * (fixture half_spread) within tolerance.
                # The fixture sets a constant half_spread = spots[0] * UNDERLYING_HALF_SPREAD_OVER_S.
                expected_tc = np.abs(underlying_trade) * expected_unit_tc
                if not np.allclose(tc_d, expected_tc, atol=1e-9, rtol=1e-9):
                    failures.append(
                        f'Delta tx_cost: observed transaction_cost {tc_d.tolist()} does not match '
                        f'|trade|*{expected_unit_tc} expectation {expected_tc.tolist()}'
                    )

    # (viii) DV vega-floor (paper §4 p.10): force the floor above the fixture's
    #        ATM vega and assert the delta-vega leg holds positions on every row
    #        (trade=0, transaction_cost=0, period_dv_held=True) and that the
    #        once-per-process RuntimeWarning fires.
    global _DV_MIN_HEDGE_VEGA, _DV_HELD_WARNED
    saved_floor = _DV_MIN_HEDGE_VEGA
    saved_warned = _DV_HELD_WARNED
    try:
        _DV_MIN_HEDGE_VEGA = 1.0e6  # well above max vega in the fixture panel
        _DV_HELD_WARNED = False
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter('always')
            summary_floor = run_daily_backtest(panel, DeterministicBacktestScenarioSource(), alpha_grid=np.array([0.0, 0.01, 0.05]))
        dv_rows_floor = summary_floor.results[summary_floor.results['strategy'] == 'delta_vega']
        if dv_rows_floor.empty:
            failures.append('DV vega-floor: no delta_vega rows emitted under forced floor')
        else:
            if 'period_dv_held' not in summary_floor.results.columns:
                failures.append('DV vega-floor: period_dv_held column missing from backtest results')
            elif not bool(dv_rows_floor['period_dv_held'].all()):
                failures.append('DV vega-floor: period_dv_held must be True on every delta_vega row under forced floor')
            trades = np.stack([np.asarray(t, dtype=float) for t in dv_rows_floor['trade'].tolist()], axis=0)
            if not np.allclose(trades, 0.0):
                failures.append('DV vega-floor: trade must be zero on held delta_vega rows')
            tx = dv_rows_floor['transaction_cost'].to_numpy(dtype=float)
            if not np.allclose(tx, 0.0):
                failures.append('DV vega-floor: transaction_cost must be zero on held delta_vega rows')
            # delta rows must NOT be flagged as held.
            delta_rows_floor = summary_floor.results[summary_floor.results['strategy'] == 'delta']
            if not delta_rows_floor.empty and bool(delta_rows_floor['period_dv_held'].any()):
                failures.append('DV vega-floor: period_dv_held must be False on delta rows')
            # lasso rows must NOT be flagged as held.
            lasso_rows_floor = summary_floor.results[summary_floor.results['strategy'] == 'lasso']
            if not lasso_rows_floor.empty and bool(lasso_rows_floor['period_dv_held'].any()):
                failures.append('DV vega-floor: period_dv_held must be False on lasso rows')
        dv_warnings = [w for w in wlog if issubclass(w.category, RuntimeWarning) and 'delta-vega benchmark held previous positions' in str(w.message)]
        if len(dv_warnings) != 1:
            failures.append(f'DV vega-floor: expected exactly one RuntimeWarning, got {len(dv_warnings)}')
    finally:
        _DV_MIN_HEDGE_VEGA = saved_floor
        _DV_HELD_WARNED = saved_warned

    # (xi) Phase 6 / r_pcp wiring: with a non-zero per-date r_pcp supplied via the
    #      panel.quotes column path, cash_drift must be non-zero on at least one
    #      row that has hedge_value > 0.  Guards against the bug where r_pcp is
    #      computed by Phase 5 M13 but never consumed by the cash-drift accumulator.
    try:
        rpcp_panel = _backtest_self_check_panel()
        rpcp_quotes = rpcp_panel.quotes.copy()
        rpcp_quotes['r_pcp'] = 0.05
        rpcp_panel_rate = HedgePanel(
            start_date=rpcp_panel.start_date,
            expiry_date=rpcp_panel.expiry_date,
            m0=rpcp_panel.m0,
            target=rpcp_panel.target,
            hedges=rpcp_panel.hedges,
            quotes=rpcp_quotes,
            missing_quotes=rpcp_panel.missing_quotes,
            trading_dates=rpcp_panel.trading_dates,
        )
        rpcp_summary = run_daily_backtest(
            rpcp_panel_rate,
            DeterministicBacktestScenarioSource(),
            alpha_grid=np.array([0.0, 0.01, 0.05]),
        )
        if 'cash_drift' not in rpcp_summary.results.columns:
            failures.append('Phase 6 r_pcp: cash_drift column missing under r_pcp fixture')
        else:
            cd = rpcp_summary.results['cash_drift'].to_numpy(dtype=float)
            if not np.any(np.abs(cd) > 1e-12):
                failures.append(
                    'Phase 6 r_pcp: cash_drift is zero across all rows under non-zero r_pcp fixture; '
                    'wiring bug remains (M9 reader not consuming r_pcp)'
                )
    except Exception as exc:  # pragma: no cover - diagnostic only
        failures.append(f'Phase 6 r_pcp: cash_drift fixture failed to run: {exc!r}')

    # (xi-b) Phase 6 / r_pcp wiring through SmoothedSurfaceMarket BS pricing.
    #        Two-date fixture with r_pcp=0.05 and r_pcp=0.06 must yield BS
    #        prices that differ from r=0 by at least 1e-3 on at least one of
    #        target_mid / hedge_mid.  Guards against the bug where
    #        SmoothedSurfaceMarket priced every BS call at r=0 because
    #        ``load_smoothed_surface_market`` defaulted ``risk_free_rate=0.0``.
    try:
        d_a = pd.Timestamp('2020-01-02')
        d_b = pd.Timestamp('2020-01-03')
        ex_d = pd.Timestamp('2020-02-01')
        spx_with_r = pd.DataFrame({
            'date': [d_a, d_b],
            'spx_close': [100.0, 100.0],
            'r_pcp': [0.05, 0.06],
        })
        spx_no_r = pd.DataFrame({
            'date': [d_a, d_b],
            'spx_close': [100.0, 100.0],
        })
        surf_rows = []
        for d in (d_a, d_b):
            for m in (0.8, 1.0, 1.2):
                for tau in (1 / 365, 30 / 365, 60 / 365):
                    surf_rows.append({
                        'date': d,
                        'moneyness': m,
                        'tau': tau,
                        'call_half_spread_over_s': 0.0005,
                        'put_half_spread_over_s': 0.0006,
                        'call_iv': 0.20 + 0.01 * m,
                        'put_iv': 0.22 + 0.01 * m,
                    })
        market_with_r = SmoothedSurfaceMarket(spx_with_r, pd.DataFrame(surf_rows))
        market_no_r = SmoothedSurfaceMarket(spx_no_r, pd.DataFrame(surf_rows))

        class _CC:
            def __init__(self, cp, k):
                self.cp_flag = cp
                self.strike = k
                self.exdate = ex_d
                self.optionid = f'{cp}{int(k)}'

        contracts = [_CC('C', 95.0), _CC('C', 100.0), _CC('C', 105.0), _CC('P', 100.0)]
        max_diff_dates = 0.0
        max_diff_vs_zero = 0.0
        for c in contracts:
            qa = market_with_r.quote_contract(d_a, c)
            qb = market_with_r.quote_contract(d_b, c)
            qz = market_no_r.quote_contract(d_a, c)
            max_diff_dates = max(max_diff_dates, abs(float(qa['mid_price']) - float(qb['mid_price'])))
            max_diff_vs_zero = max(max_diff_vs_zero, abs(float(qa['mid_price']) - float(qz['mid_price'])))
        if max_diff_dates < 1e-3:
            failures.append(
                f'Phase 6 r_pcp BS: max BS mid_price diff across dates with r_pcp=0.05 vs 0.06 '
                f'was only {max_diff_dates:.3e} (< 1e-3); rate not entering BS pricing'
            )
        if max_diff_vs_zero < 1e-3:
            failures.append(
                f'Phase 6 r_pcp BS: max BS mid_price diff vs r=0 baseline was only '
                f'{max_diff_vs_zero:.3e} (< 1e-3); surface BS still using r=0'
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        failures.append(f'Phase 6 r_pcp BS: fixture failed to run: {exc!r}')

    # (xii) Phase 4 M9-psi: paper-shaped fixture for self-financing ψ_t (paper Eq 4-5).
    #       With Π_t = target_mid = 100, hedge_value = 20, transaction_cost = 0,
    #       r = 0.05, dt = 1/252:
    #         ψ_t       = 100 - 20 - 0 = 80
    #         cash_drift = −ψ_t · r · dt = −80 · 0.05 / 252 = −0.015873015873...
    #       Old (broken) proxy ψ ≈ −hedge_value would give +20·0.05/252 ≈ +0.003968.
    try:
        psi_fixture = _daily_result(
            start_date=pd.Timestamp('2020-01-02'),
            end_date=pd.Timestamp('2020-01-03'),
            strategy='lasso',
            positions=np.array([1.0], dtype=float),
            previous_positions=np.array([1.0], dtype=float),
            alpha=None,
            target_change=0.0,
            hedge_changes=np.array([0.0], dtype=float),
            target_delta=0.0,
            target_vega=0.0,
            hedge_delta=np.array([0.0], dtype=float),
            hedge_vega=np.array([0.0], dtype=float),
            half_spreads=np.array([0.0], dtype=float),
            current_hedge_mid=np.array([20.0], dtype=float),
            rate=0.05,
            dt=1.0 / 252.0,
            period_start=pd.Timestamp('2020-01-02'),
            previous_cumulative_z=0.0,
            pi_t=100.0,
        )
        expected_psi_cash_drift = -(100.0 - 20.0 - 0.0) * 0.05 * (1.0 / 252.0)
        if abs(psi_fixture.cash_drift - expected_psi_cash_drift) > 1e-12:
            failures.append(
                f'Phase 4 M9-psi: paper-shaped cash_drift mismatch; '
                f'expected {expected_psi_cash_drift!r}, got {psi_fixture.cash_drift!r}'
            )
        # Π_{t+Δt} = Π_t + ψ_t r Δt + Σφ ΔH = 100 + 80·0.05/252 + 0 = 100.015873...
        expected_pi_next = 100.0 + (100.0 - 20.0 - 0.0) * 0.05 * (1.0 / 252.0) + 0.0
        if abs(psi_fixture.pi_t_next - expected_pi_next) > 1e-12:
            failures.append(
                f'Phase 4 M9-psi: paper-shaped pi_t_next mismatch; '
                f'expected {expected_pi_next!r}, got {psi_fixture.pi_t_next!r}'
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        failures.append(f'Phase 4 M9-psi: paper-shaped fixture failed to run: {exc!r}')

    return failures


def paper_output_self_check() -> list[str]:
    """Run deterministic checks for paper-level reporting helpers."""

    failures = []
    panel = _backtest_self_check_panel()
    summary = run_daily_backtest(panel, DeterministicBacktestScenarioSource(), alpha_grid=np.array([0.0, 0.01, 0.05]))
    expected_strategies = {'lasso', 'delta', 'delta_vega'}

    helper_frames = {
        'tracking_error_summary': tracking_error_summary(summary),
        'transaction_cost_summary': transaction_cost_summary(summary),
        'selected_hedge_count_turnover_summary': selected_hedge_count_turnover_summary(summary),
        'greek_residual_summary': greek_residual_summary(summary),
        'strategy_comparison_table': strategy_comparison_table(summary),
    }
    skipped = skip_summary(summary)

    for name, frame in helper_frames.items():
        strategies = set(frame.get('strategy', []))
        if strategies != expected_strategies:
            failures.append(f'{name} strategy rows {strategies}, expected {expected_strategies}')

    if skipped.empty or 'missing_quote' not in set(skipped['reason']):
        failures.append('skip summary did not preserve missing_quote reason')
    elif int(skipped['total_unique_skipped_intervals'].max()) != summary.skipped_interval_count:
        failures.append('skip summary total unique skipped intervals disagrees with BacktestSummary')

    if summary.results.empty:
        failures.append('paper output self-check fixture produced no backtest results')
        return failures

    required_result_columns = {
        'target_delta',
        'target_vega',
        'hedge_delta_exposure',
        'hedge_vega_exposure',
        'delta_residual',
        'vega_residual',
    }
    missing_columns = sorted(required_result_columns.difference(summary.results.columns))
    if missing_columns:
        failures.append(f'missing Greek exposure result columns: {missing_columns}')
        return failures

    for name, frame in helper_frames.items():
        for column in frame.columns:
            if column == 'strategy':
                continue
            values = pd.to_numeric(frame[column], errors='coerce')
            if values.notna().any() and not np.all(np.isfinite(values.dropna().to_numpy(dtype=float))):
                failures.append(f'{name}.{column} contains non-finite values')

    if np.any(pd.to_numeric(summary.results['transaction_cost'], errors='coerce').to_numpy(dtype=float) < -1e-12):
        failures.append('transaction costs must be nonnegative')

    activity = helper_frames['selected_hedge_count_turnover_summary'].set_index('strategy')
    for strategy, group in summary.results.groupby('strategy', sort=False):
        positions = [_one_dimensional_array(value) for value in group['positions']]
        trades = [_one_dimensional_array(value) for value in group['trade']]
        expected_nonzero_positions = float(np.mean([np.count_nonzero(np.abs(value) > 1e-12) for value in positions]))
        expected_nonzero_trades = float(np.mean([np.count_nonzero(np.abs(value) > 1e-12) for value in trades]))
        expected_turnover = float(np.sum([np.sum(np.abs(value)) for value in trades]))
        expected_gross = float(np.mean([np.sum(np.abs(value)) for value in positions]))
        row = activity.loc[strategy]
        if not np.isclose(float(row['nonzero_positions']), expected_nonzero_positions):
            failures.append(f'{strategy} nonzero position count inconsistent with stored positions')
        if not np.isclose(float(row['nonzero_trades']), expected_nonzero_trades):
            failures.append(f'{strategy} nonzero trade count inconsistent with stored trades')
        if not np.isclose(float(row['l1_turnover']), expected_turnover):
            failures.append(f'{strategy} L1 turnover inconsistent with stored trades')
        if not np.isclose(float(row['gross_position']), expected_gross):
            failures.append(f'{strategy} gross position inconsistent with stored positions')

    for _, row in summary.results.iterrows():
        if not np.isclose(float(row['delta_residual']), float(row['target_delta']) - float(row['hedge_delta_exposure'])):
            failures.append(f"{row['strategy']} delta residual inconsistent with stored exposure")
        if not np.isclose(float(row['vega_residual']), float(row['target_vega']) - float(row['hedge_vega_exposure'])):
            failures.append(f"{row['strategy']} vega residual inconsistent with stored exposure")

    complete_start = panel.trading_dates[0]
    complete_end = panel.trading_dates[1]
    current_target, current_hedges, _, _, missing = _complete_interval_quotes(panel, complete_start, complete_end)
    if not missing.empty:
        failures.append('paper output fixture first interval should be complete')
        return failures
    hedge_delta = pd.to_numeric(current_hedges['delta'], errors='coerce').to_numpy(dtype=float)
    hedge_vega = pd.to_numeric(current_hedges['vega'], errors='coerce').to_numpy(dtype=float)

    delta_rows = summary.results[summary.results['strategy'] == 'delta']
    if np.linalg.matrix_rank(hedge_delta.reshape(1, -1)) == 1:
        if not np.allclose(delta_rows['delta_residual'].to_numpy(dtype=float), 0.0, atol=1e-10):
            failures.append('delta strategy should have near-zero delta residual in full-rank fixture')

    delta_vega_rows = summary.results[summary.results['strategy'] == 'delta_vega']
    # Vega-floor (paper §4 p.10): when delta-vega held its previous positions
    # because |atm_vega| < _DV_MIN_HEDGE_VEGA, residuals are NOT expected to be
    # zero — exclude those rows from the full-rank residual assertion.
    if 'period_dv_held' in delta_vega_rows.columns:
        delta_vega_rows_rebal = delta_vega_rows[~delta_vega_rows['period_dv_held'].astype(bool)]
    else:
        delta_vega_rows_rebal = delta_vega_rows
    if np.linalg.matrix_rank(np.vstack([hedge_delta, hedge_vega])) == 2 and not delta_vega_rows_rebal.empty:
        if not np.allclose(delta_vega_rows_rebal['delta_residual'].to_numpy(dtype=float), 0.0, atol=1e-10):
            failures.append('delta-vega strategy should have near-zero delta residual in full-rank fixture')
        if not np.allclose(delta_vega_rows_rebal['vega_residual'].to_numpy(dtype=float), 0.0, atol=1e-10):
            failures.append('delta-vega strategy should have near-zero vega residual in full-rank fixture')

    lasso_rows = summary.results[summary.results['strategy'] == 'lasso']
    if not np.all(np.isfinite(lasso_rows[['delta_residual', 'vega_residual']].to_numpy(dtype=float))):
        failures.append('lasso residuals must be finite')
    residual_summary = helper_frames['greek_residual_summary'].set_index('strategy')
    if 'lasso' in residual_summary.index and not lasso_rows.empty:
        lasso_delta = lasso_rows['delta_residual'].to_numpy(dtype=float)
        lasso_vega = lasso_rows['vega_residual'].to_numpy(dtype=float)
        if not np.isclose(float(residual_summary.loc['lasso', 'mean_abs_delta_residual']), float(np.mean(np.abs(lasso_delta)))):
            failures.append('lasso delta residual summary inconsistent with stored residuals')
        if not np.isclose(float(residual_summary.loc['lasso', 'vega_residual_rmse']), _rmse(lasso_vega)):
            failures.append('lasso vega residual RMSE inconsistent with stored residuals')

    return failures

def surface_market_self_check(processed_dir: Path = Path("data/processed_shared_grid_11x9")) -> list[str]:
    """Run deterministic checks for surface-valued quote lookup and backtesting."""

    failures = []
    try:
        market = load_smoothed_surface_market(processed_dir)
    except Exception as exc:
        return [f"failed to load smoothed surface market: {exc!r}"]

    try:
        date = market.price_surfaces["date"].min()
        rows = market.price_surfaces[market.price_surfaces["date"] == date]
        grid_row = rows.sort_values(["moneyness", "tau"]).iloc[0]
        value = market.interpolate_value(date, "call_iv", float(grid_row["moneyness"]), float(grid_row["tau"]))
        if not np.isclose(value, float(grid_row["call_iv"])):
            failures.append("surface exact-grid lookup did not return the grid value")
        linear = np.array([[100.0 * m + 10.0 * tau for tau in market.tau_grid] for m in market.m_grid])
        saved = market._surface_cache.get(_as_timestamp(date), {}).copy()
        market._surface_cache[_as_timestamp(date)] = {"call_iv": linear}
        actual = market.interpolate_value(date, "call_iv", 0.975, 0.10)
        expected = 100.0 * 0.975 + 10.0 * 0.10
        if not np.isclose(actual, expected):
            failures.append(f"linear interpolation convention mismatch: {actual} != {expected}")
        if saved:
            market._surface_cache[_as_timestamp(date)] = saved
        else:
            market._surface_cache.pop(_as_timestamp(date), None)
    except Exception as exc:
        failures.append(f"surface interpolation fixture failed: {exc!r}")

    try:
        panel = _backtest_self_check_panel()
        # Phase 3: panel has 5 dates with drifting spots; mirror them here.
        market_dates = pd.DataFrame({
            "date": panel.trading_dates,
            "spx_close": [100.0, 102.0, 108.0, 113.0, 115.0],
        })
        surface_rows = []
        for date in panel.trading_dates:
            for m in (0.8, 1.0, 1.2):
                for tau in (1 / 365, 30 / 365, 60 / 365):
                    surface_rows.append({
                        "date": date,
                        "moneyness": m,
                        "tau": tau,
                        "call_half_spread_over_s": 0.0005,
                        "put_half_spread_over_s": 0.0006,
                        "call_iv": 0.20 + 0.01 * m,
                        "put_iv": 0.22 + 0.01 * m,
                    })
        fixture_market = SmoothedSurfaceMarket(market_dates, pd.DataFrame(surface_rows))
        summary = run_daily_backtest(panel, DeterministicBacktestScenarioSource(), alpha_grid=np.array([0.0, 0.01]), quote_source=fixture_market)
        if summary.skipped_interval_count != 0:
            failures.append(f"surface backtest should not skip missing optionid quotes, found {summary.skipped_interval_count}")
        # Phase 3: 4 surface intervals \u00d7 3 strategies = 12 rows.
        if summary.results.shape[0] != 12:
            failures.append(f"surface backtest expected twelve strategy rows for four intervals, found {summary.results.shape[0]}")
        if not summary.results.empty and not np.all(np.isfinite(summary.results[["target_change", "hedge_change", "transaction_cost"]].to_numpy(dtype=float))):
            failures.append("surface backtest produced non-finite realized values or costs")
        if not summary.results.empty and np.any(summary.results["transaction_cost"].to_numpy(dtype=float) < -1e-12):
            failures.append("surface transaction costs must be nonnegative")
    except Exception as exc:
        failures.append(f"surface backtest fixture failed: {exc!r}")

    # Phase 6 / r_pcp wiring: confirm SmoothedSurfaceMarket builds a per-date
    # rate lookup and that BS pricing in ``quote_contract`` actually varies
    # with r_pcp.  Bug guard: prior implementation ignored r_pcp and priced at
    # r=0 for every date.
    try:
        date_a = pd.Timestamp("2020-01-02")
        date_b = pd.Timestamp("2020-01-03")
        rpcp_spx = pd.DataFrame(
            {
                "date": [date_a, date_b],
                "spx_close": [100.0, 100.0],
                "r_pcp": [0.05, 0.06],
            }
        )
        rpcp_rows = []
        for d in (date_a, date_b):
            for m in (0.8, 1.0, 1.2):
                for tau in (1 / 365, 30 / 365, 60 / 365):
                    rpcp_rows.append(
                        {
                            "date": d,
                            "moneyness": m,
                            "tau": tau,
                            "call_half_spread_over_s": 0.0005,
                            "put_half_spread_over_s": 0.0006,
                            "call_iv": 0.20 + 0.01 * m,
                            "put_iv": 0.22 + 0.01 * m,
                        }
                    )
        rpcp_market = SmoothedSurfaceMarket(rpcp_spx, pd.DataFrame(rpcp_rows))
        if not rpcp_market._rate_by_date:
            failures.append("Phase 6 r_pcp: SmoothedSurfaceMarket._rate_by_date empty under r_pcp fixture")
        if abs(rpcp_market._rate_for(date_a) - 0.05) > 1e-12:
            failures.append(
                f"Phase 6 r_pcp: _rate_for(date_a) expected 0.05, got {rpcp_market._rate_for(date_a)!r}"
            )
        if abs(rpcp_market._rate_for(date_b) - 0.06) > 1e-12:
            failures.append(
                f"Phase 6 r_pcp: _rate_for(date_b) expected 0.06, got {rpcp_market._rate_for(date_b)!r}"
            )

        class _C:
            def __init__(self, cp_flag, strike, exdate, optionid, hedge_moneyness=None):
                self.cp_flag = cp_flag
                self.strike = strike
                self.exdate = exdate
                self.optionid = optionid
                if hedge_moneyness is not None:
                    self.hedge_moneyness = hedge_moneyness

        exdate = pd.Timestamp("2020-02-01")
        c_atm = _C("C", 100.0, exdate, 1)
        q_a = rpcp_market.quote_contract(date_a, c_atm)
        q_b = rpcp_market.quote_contract(date_b, c_atm)
        price_diff = abs(float(q_a["mid_price"]) - float(q_b["mid_price"]))
        if price_diff < 1e-3:
            failures.append(
                f"Phase 6 r_pcp: BS mid_price between r=0.05 and r=0.06 differs by only {price_diff:.3e} "
                "(< 1e-3); rate not flowing through quote_contract"
            )
        # Compare against the broken r=0 baseline: build a market with no r_pcp
        # column (constructor default r=0).
        rpcp_spx_no_r = pd.DataFrame({"date": [date_a], "spx_close": [100.0]})
        rpcp_rows_no_r = [r for r in rpcp_rows if r["date"] == date_a]
        zero_market = SmoothedSurfaceMarket(rpcp_spx_no_r, pd.DataFrame(rpcp_rows_no_r))
        q_zero = zero_market.quote_contract(date_a, c_atm)
        diff_vs_zero = abs(float(q_a["mid_price"]) - float(q_zero["mid_price"]))
        if diff_vs_zero < 1e-3:
            failures.append(
                f"Phase 6 r_pcp: r=0.05 mid_price differs from r=0 baseline by only {diff_vs_zero:.3e} "
                "(< 1e-3); BS pricing still ignoring r_pcp"
            )
        if zero_market.rate_lookup_misses == 0:
            failures.append(
                "Phase 6 r_pcp: rate_lookup_misses must increment when r_pcp column absent"
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        failures.append(f"Phase 6 r_pcp: surface-market r_pcp fixture failed: {exc!r}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--start-date")
    parser.add_argument("--m0", type=float)
    parser.add_argument("--target-days", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--solver-self-check", action="store_true")
    parser.add_argument("--scenario-adapter-self-check", action="store_true")
    parser.add_argument("--backtest-self-check", action="store_true")
    parser.add_argument("--paper-output-self-check", action="store_true")
    parser.add_argument("--surface-market-self-check", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_shared_grid_11x9"))
    args = parser.parse_args()

    if args.solver_self_check:
        violations = solver_self_check()
        if violations:
            print("SOLVER_SELF_CHECK=FAIL")
            for violation in violations:
                print(f"- {violation}")
            return 1
        print("SOLVER_SELF_CHECK=PASS")
        return 0

    if args.scenario_adapter_self_check:
        violations = scenario_adapter_self_check()
        if violations:
            print("SCENARIO_ADAPTER_SELF_CHECK=FAIL")
            for violation in violations:
                print(f"- {violation}")
            return 1
        print("SCENARIO_ADAPTER_SELF_CHECK=PASS")
        return 0

    if args.backtest_self_check:
        violations = backtest_self_check()
        if violations:
            print("BACKTEST_SELF_CHECK=FAIL")
            for violation in violations:
                print(f"- {violation}")
            return 1
        print("BACKTEST_SELF_CHECK=PASS")
        return 0

    if args.paper_output_self_check:
        violations = paper_output_self_check()
        if violations:
            print("PAPER_OUTPUT_SELF_CHECK=FAIL")
            for violation in violations:
                print(f"- {violation}")
            return 1
        print("PAPER_OUTPUT_SELF_CHECK=PASS")
        return 0

    if args.surface_market_self_check:
        violations = surface_market_self_check(args.processed_dir)
        if violations:
            print("SURFACE_MARKET_SELF_CHECK=FAIL")
            for violation in violations:
                print(f"- {violation}")
            return 1
        print("SURFACE_MARKET_SELF_CHECK=PASS")
        return 0

    if args.start_date is None or args.m0 is None:
        parser.error(
            "--start-date and --m0 are required unless --solver-self-check, "
            "--scenario-adapter-self-check, --backtest-self-check, "
            "--paper-output-self-check, or --surface-market-self-check is set"
        )

    panel = build_instrument_panel(
        start_date=args.start_date,
        m0=args.m0,
        data_dir=args.data_dir,
        target_days=args.target_days,
    )
    violations = panel_self_check(panel)
    print(f"start_date={panel.start_date.date()}")
    print(f"expiry_date={panel.expiry_date.date()}")
    print(f"m0={panel.m0}")
    print(f"target_contracts={len(panel.target)}")
    print(f"hedge_contracts={len(panel.hedges)}")
    print(f"observed_quote_rows={len(panel.quotes)}")
    print(f"trading_dates={len(panel.trading_dates)}")
    print("missing_quote_summary:")
    print(panel.missing_quotes.to_string(index=False))

    if args.output_dir:
        _write_outputs(panel, args.output_dir)
        print(f"wrote_outputs={args.output_dir}")

    if violations:
        print("SELF_CHECK=FAIL")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("SELF_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
