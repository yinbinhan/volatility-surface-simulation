#!/usr/bin/env python3
"""Instrument-panel utilities for data-driven hedging experiments.

Phase 1 is deliberately limited to observed OptionMetrics quotes. The raw
option files are volume-filtered, so selected instruments may be missing on
some trading dates. This module reports those gaps and does not fill them.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path("data/optionmetrics_spx_20000103_20230228")
HEDGE_MONEYNESS = (0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1)
UNDERLYING_OPTIONID = "UNDERLYING_SPX"
UNDERLYING_CP_FLAG = "U"
UNDERLYING_HALF_SPREAD = 0.0
BENCHMARK_VEGA_FLOOR = 1e-3
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
    fixed_atm_optionid: object | None = None


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
) -> pd.DataFrame:
    """Select paper-style candidate hedging instruments on the start date."""

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
    return hedges.drop_duplicates("optionid").reset_index(drop=True)


def _underlying_metadata_row(start_spot: float, expiry: pd.Timestamp, actual_start: pd.Timestamp) -> dict[str, object]:
    return {
        "role": "underlying",
        "optionid": UNDERLYING_OPTIONID,
        "cp_flag": UNDERLYING_CP_FLAG,
        "strike": float(start_spot),
        "strike_price": float(start_spot),
        "spot": float(start_spot),
        "moneyness": 1.0,
        "mid_price": float(start_spot),
        "half_spread": UNDERLYING_HALF_SPREAD,
        "bid_ask_spread": 2.0 * UNDERLYING_HALF_SPREAD,
        "delta": 1.0,
        "vega": 0.0,
        "impl_volatility": np.nan,
        "days_to_exp": int(max((expiry - actual_start).days, 0)),
        "ttm": max((expiry - actual_start).days, 0) / 365.0,
        "exdate": expiry,
        "date": actual_start,
        "hedge_moneyness": 1.0,
        "volume": np.nan,
        "open_interest": np.nan,
        "symbol": UNDERLYING_OPTIONID,
    }


def _append_underlying_hedge(
    option_hedges: pd.DataFrame, start_spot: float, expiry: pd.Timestamp, actual_start: pd.Timestamp
) -> pd.DataFrame:
    underlying_row = pd.DataFrame([_underlying_metadata_row(start_spot, expiry, actual_start)])
    return pd.concat([option_hedges, underlying_row], ignore_index=True, sort=False)


def _underlying_quote_rows(underlying: pd.DataFrame, expiry: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for row in underlying.itertuples(index=False):
        date = _as_timestamp(row.date)
        spot = float(row.close)
        rows.append(
            {
                "date": date,
                "exdate": expiry,
                "optionid": UNDERLYING_OPTIONID,
                "role": "underlying",
                "cp_flag": UNDERLYING_CP_FLAG,
                "strike": spot,
                "strike_price": spot,
                "spot": spot,
                "moneyness": 1.0,
                "mid_price": spot,
                "half_spread": UNDERLYING_HALF_SPREAD,
                "bid_ask_spread": 2.0 * UNDERLYING_HALF_SPREAD,
                "delta": 1.0,
                "vega": 0.0,
                "impl_volatility": np.nan,
                "days_to_exp": int(max((expiry - date).days, 0)),
                "ttm": max((expiry - date).days, 0) / 365.0,
                "volume": np.nan,
                "open_interest": np.nan,
                "symbol": UNDERLYING_OPTIONID,
            }
        )
    return pd.DataFrame(rows)


def _is_underlying_frame(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    if "optionid" in frame.columns:
        mask = mask | frame["optionid"].map(lambda value: str(value) == UNDERLYING_OPTIONID)
    if "cp_flag" in frame.columns:
        mask = mask | frame["cp_flag"].astype(str).eq(UNDERLYING_CP_FLAG)
    if "role" in frame.columns:
        mask = mask | frame["role"].astype(str).eq("underlying")
    return mask


def _select_fixed_atm_optionid(hedges: pd.DataFrame, start_spot: float) -> object | None:
    option_hedges = hedges.loc[~_is_underlying_frame(hedges)].copy()
    if option_hedges.empty:
        return None
    option_hedges["_atm_distance"] = (pd.to_numeric(option_hedges["strike"], errors="coerce") - float(start_spot)).abs()
    option_hedges = option_hedges.sort_values(["_atm_distance", "strike", "optionid"])
    return option_hedges.iloc[0]["optionid"]


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
    option_hedges = select_hedge_candidates(start_rows, start_spot, expiry, target_ids)
    target["optionid"] = target["optionid"].map(str)
    option_hedges["optionid"] = option_hedges["optionid"].map(str)
    fixed_atm_optionid = _select_fixed_atm_optionid(option_hedges, start_spot)
    hedges = _append_underlying_hedge(option_hedges, start_spot, expiry, actual_start)
    selected = pd.concat([target, hedges], ignore_index=True, sort=False)
    selected["optionid"] = selected["optionid"].map(str)

    underlying = load_underlying(data_dir=data_dir, start_date=actual_start, end_date=expiry)
    trading_dates = expected_trading_dates(underlying, actual_start, expiry)
    options = load_raw_options(data_dir=data_dir, start_date=actual_start, end_date=expiry)
    options["optionid"] = options["optionid"].map(str)
    quotes = options[options["optionid"].isin(selected["optionid"])].copy()
    quotes = quotes.merge(
        selected[["optionid", "role"]].drop_duplicates(),
        on="optionid",
        how="left",
        validate="many_to_one",
    )
    underlying_quotes = _underlying_quote_rows(underlying, expiry)
    quotes = pd.concat([quotes, underlying_quotes], ignore_index=True, sort=False)
    quotes = quotes.sort_values(["date", "role", "cp_flag", "strike"]).reset_index(drop=True)
    missing = quote_coverage(quotes, selected, trading_dates)

    return HedgePanel(
        start_date=actual_start,
        expiry_date=expiry,
        m0=m0,
        target=target,
        hedges=hedges,
        quotes=quotes,
        missing_quotes=missing,
        trading_dates=trading_dates,
        fixed_atm_optionid=fixed_atm_optionid,
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
    if _is_underlying_frame(panel.hedges).sum() != 1:
        violations.append("hedge candidate set must contain exactly one underlying row")
    if panel.fixed_atm_optionid is None:
        violations.append("fixed inception ATM benchmark option id is missing")
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


def _partition_hedge_contracts_for_surface(hedge_contracts: pd.DataFrame, cp_col: str, strike_col: str, tau_col: str):
    hedge_contracts = pd.DataFrame(hedge_contracts).reset_index(drop=True)
    underlying_mask = _is_underlying_frame(hedge_contracts)
    underlying_indices = np.flatnonzero(underlying_mask.to_numpy(dtype=bool))
    if underlying_indices.size > 1:
        raise ValueError("surface scenario adapters support at most one underlying hedge row")
    option_indices = np.flatnonzero(~underlying_mask.to_numpy(dtype=bool))
    option_contracts_raw = hedge_contracts.iloc[option_indices].reset_index(drop=True)
    option_contracts = _validate_contracts(option_contracts_raw, "hedge_contracts", cp_col, strike_col, tau_col)
    return hedge_contracts, option_contracts, option_indices, underlying_indices


def _assemble_hedge_next_values(
    hedge_count: int,
    option_indices: np.ndarray,
    option_next_values: np.ndarray,
    underlying_indices: np.ndarray,
    spots: np.ndarray,
) -> np.ndarray:
    next_hedge = np.empty((len(spots), hedge_count), dtype=float)
    if len(option_indices):
        next_hedge[:, option_indices] = option_next_values
    if len(underlying_indices):
        next_hedge[:, int(underlying_indices[0])] = spots
    return next_hedge


def adapt_normalized_price_surface(scenarios: NormalizedPriceSurfaceScenarios) -> SolverScenarioArrays:
    """Revalue selected contracts from normalized price surfaces."""

    target_contracts = _validate_contracts(scenarios.target_contracts, "target_contracts", scenarios.cp_col, scenarios.strike_col, scenarios.tau_col)
    hedge_contracts, hedge_option_contracts, option_indices, underlying_indices = _partition_hedge_contracts_for_surface(
        scenarios.hedge_contracts, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col
    )
    current_target, current_hedge = _validate_surface_contract_counts(target_contracts, hedge_contracts, scenarios.current_target_values, scenarios.current_hedge_values)
    _require_columns(scenarios.normalized_surface, [scenarios.scenario_col, scenarios.cp_col, scenarios.moneyness_col, scenarios.tau_col, scenarios.normalized_price_col], "normalized_surface")
    scenario_ids, spots = _scenario_ids_and_spots(scenarios.normalized_surface, scenarios.scenario_col, scenarios.spot_next)
    next_target = _revalue_normalized_surface(target_contracts, scenarios.normalized_surface, scenario_ids, spots, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.normalized_price_col)
    option_next = _revalue_normalized_surface(hedge_option_contracts, scenarios.normalized_surface, scenario_ids, spots, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.normalized_price_col)
    next_hedge = _assemble_hedge_next_values(len(hedge_contracts), option_indices, option_next, underlying_indices, spots)
    return adapt_selected_instrument_values(SelectedInstrumentValueScenarios(current_target, current_hedge, next_target, next_hedge))


def adapt_iv_surface(scenarios: IVSurfaceScenarios) -> SolverScenarioArrays:
    """Revalue selected contracts from IV surfaces with Black-Scholes."""

    target_contracts = _validate_contracts(scenarios.target_contracts, "target_contracts", scenarios.cp_col, scenarios.strike_col, scenarios.tau_col)
    hedge_contracts, hedge_option_contracts, option_indices, underlying_indices = _partition_hedge_contracts_for_surface(
        scenarios.hedge_contracts, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col
    )
    current_target, current_hedge = _validate_surface_contract_counts(target_contracts, hedge_contracts, scenarios.current_target_values, scenarios.current_hedge_values)
    _require_columns(scenarios.iv_surface, [scenarios.scenario_col, scenarios.cp_col, scenarios.moneyness_col, scenarios.tau_col, scenarios.iv_col], "iv_surface")
    scenario_ids, spots = _scenario_ids_and_spots(scenarios.iv_surface, scenarios.scenario_col, scenarios.spot_next)
    next_target = _revalue_iv_surface(target_contracts, scenarios.iv_surface, scenario_ids, spots, scenarios.risk_free_rate, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.iv_col)
    option_next = _revalue_iv_surface(hedge_option_contracts, scenarios.iv_surface, scenario_ids, spots, scenarios.risk_free_rate, scenarios.cp_col, scenarios.strike_col, scenarios.tau_col, scenarios.scenario_col, scenarios.moneyness_col, scenarios.iv_col)
    next_hedge = _assemble_hedge_next_values(len(hedge_contracts), option_indices, option_next, underlying_indices, spots)
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
    hedge_contracts = pd.DataFrame(
        {
            "optionid": [201, 202, UNDERLYING_OPTIONID],
            "role": ["hedge", "hedge", "underlying"],
            "cp_flag": ["C", "P", UNDERLYING_CP_FLAG],
            "strike": [95.0, 105.0, 100.0],
            "tau": [0.08, 0.08, np.nan],
        }
    )
    rows = []
    for scenario_id, shift in [(0, 0.0), (1, 0.01)]:
        for cp_flag in ["C", "P"]:
            for moneyness in [0.95, 1.0, 1.05]:
                base = 0.10 if cp_flag == "C" else 0.06
                rows.append({"scenario_id": scenario_id, "cp_flag": cp_flag, "moneyness": moneyness, "tau": 0.08, "normalized_price": base + shift + 0.2 * abs(moneyness - 1.0), "implied_volatility": 0.20 + shift})
    surface = pd.DataFrame(rows)
    normalized = adapt_scenarios_to_solver(
        NormalizedPriceSurfaceScenarios(
            target_contracts, hedge_contracts, np.zeros(2), np.array([0.0, 0.0, 100.0]), surface, np.array([100.0, 100.0])
        )
    )
    if normalized.target_changes.shape != (2,) or normalized.hedge_changes.shape != (2, 3):
        failures.append("normalized surface revaluation returned wrong solver array shapes")
    if not np.all(np.isfinite(normalized.target_changes)) or not np.all(np.isfinite(normalized.hedge_changes)):
        failures.append("normalized surface revaluation returned non-finite arrays")
    if not np.allclose(normalized.hedge_changes[0], np.array([11.0, 7.0, 0.0])):
        failures.append("hedge contract rows did not align with hedge matrix columns")
    if not np.allclose(normalized.hedge_changes[:, 2], np.array([0.0, 0.0])):
        failures.append("normalized surface underlying column did not equal spot changes")

    iv = adapt_scenarios_to_solver(
        IVSurfaceScenarios(
            target_contracts, hedge_contracts, np.zeros(2), np.array([0.0, 0.0, 100.0]), surface, np.array([100.0, 101.0]), risk_free_rate=0.0
        )
    )
    if iv.target_changes.shape != (2,) or iv.hedge_changes.shape != (2, 3):
        failures.append("IV surface revaluation returned wrong solver array shapes")
    if not np.all(np.isfinite(iv.target_changes)) or not np.all(np.isfinite(iv.hedge_changes)):
        failures.append("IV surface revaluation returned non-finite arrays")
    if not np.allclose(iv.hedge_changes[:, 2], np.array([0.0, 1.0])):
        failures.append("IV surface underlying column did not equal spot changes")

    expect_failure("invalid direct shape", lambda: adapt_scenarios_to_solver(DirectScenarioChanges(np.ones(2), np.ones((3, 1)))))
    expect_failure("missing spot", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.array([0.0, 0.0, 100.0]), surface, pd.Series({0: 100.0}))))
    bad_tau_contracts = target_contracts.copy()
    bad_tau_contracts.loc[0, "tau"] = 0.0
    expect_failure("nonpositive tau", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(bad_tau_contracts, hedge_contracts, np.zeros(2), np.array([0.0, 0.0, 100.0]), surface, np.array([100.0, 100.0]))))
    call_only_surface = surface[surface["cp_flag"] == "C"].copy()
    expect_failure("missing put/call route", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.array([0.0, 0.0, 100.0]), call_only_surface, np.array([100.0, 100.0]))))
    expect_failure("column/instrument mismatch", lambda: adapt_scenarios_to_solver(NormalizedPriceSurfaceScenarios(target_contracts, hedge_contracts, np.zeros(2), np.zeros(1), surface, np.array([100.0, 100.0]))))

    return failures



@dataclass(frozen=True)
class TransactionCostLassoResult:
    """Solution summary for the transaction-cost lasso hedge update."""

    phi: np.ndarray
    trade: np.ndarray
    alpha: float
    intercept: float  # A_t from paper eq.(20): unpenalized, estimated jointly with phi
    g0_scale: float   # g_0 from paper eq.(20): initial gross position = V_t
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
    g0_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    y = _as_float_vector(target_changes, "target_changes")
    x = _as_float_matrix(hedge_changes, "hedge_changes")
    phi_prev_arr = _as_float_vector(phi_prev, "phi_prev")
    costs = _as_float_vector(c_i, "c_i")
    alpha = float(alpha)
    g0_scale = float(g0_scale)
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
    if not np.isfinite(g0_scale) or g0_scale <= 0:
        raise ValueError("g0_scale must be a positive finite scalar")
    return y, x, phi_prev_arr, costs, alpha, g0_scale


def _transaction_cost_lasso_components(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi: np.ndarray | Iterable[float],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0_scale: float,
    intercept: float = 0.0,
) -> tuple[float, float, float]:
    # Paper eq.(20): (1/N) sum_k [DV^k - A_t - sum_i phi_i DH_i^k]^2 + alpha*g0*sum_i c_i|trade_i|
    y, x, phi_prev_arr, costs, alpha, g0_scale = _validate_lasso_inputs(
        target_changes, hedge_changes, phi_prev, c_i, alpha, g0_scale
    )
    phi_arr = _as_float_vector(phi, "phi")
    if phi_arr.shape[0] != phi_prev_arr.shape[0]:
        raise ValueError("phi length must match phi_prev length")

    residual = y - float(intercept) - x @ phi_arr
    fit_loss = float(np.mean(residual * residual))
    transaction_penalty = float(alpha * g0_scale * np.sum(costs * np.abs(phi_arr - phi_prev_arr)))
    objective_value = fit_loss + transaction_penalty
    return objective_value, fit_loss, transaction_penalty


def transaction_cost_lasso_objective(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi: np.ndarray | Iterable[float],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0_scale: float = 1.0,
    intercept: float = 0.0,
) -> float:
    """Return MSE plus weighted L1 transaction cost on trade increments.

    Implements paper eq.(20): (1/N) sum_k [DV^k - intercept - sum_i phi_i DH_i^k]^2
    + alpha * g0_scale * sum_i c_i * |phi_i - phi_prev_i|.
    intercept is A_t (estimated jointly); g0_scale is g_0 = V_t.
    """

    objective_value, _, _ = _transaction_cost_lasso_components(
        target_changes, hedge_changes, phi, phi_prev, c_i, alpha, g0_scale, intercept
    )
    return objective_value


def solve_transaction_cost_lasso(
    target_changes: np.ndarray | Iterable[float],
    hedge_changes: np.ndarray | Iterable[Iterable[float]],
    phi_prev: np.ndarray | Iterable[float],
    c_i: np.ndarray | Iterable[float],
    alpha: float,
    g0_scale: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> TransactionCostLassoResult:
    """Solve the transaction-cost lasso hedge update by coordinate descent.

    Implements paper eq.(20). The intercept A_t is estimated jointly with phi
    (unpenalized coordinate). g0_scale = g_0 = V_t scales the L1 penalty so
    that alpha is dimensionless regardless of portfolio size (paper §3).
    lambdas use a 1/2 factor because the objective coefficient is 1/N (not
    1/2N), giving coordinate updates consistent with the paper's formulation.
    """

    y, x, phi_prev_arr, costs, alpha, g0_scale = _validate_lasso_inputs(
        target_changes, hedge_changes, phi_prev, c_i, alpha, g0_scale
    )
    max_iter = int(max_iter)
    tol = float(tol)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if tol < 0 or not np.isfinite(tol):
        raise ValueError("tol must be a finite nonnegative scalar")

    n_scenarios, n_hedges = x.shape
    trade = np.zeros(n_hedges, dtype=float)
    # Initialize A_t (intercept) at the optimal value for phi = phi_prev.
    intercept = float(np.mean(y - x @ phi_prev_arr))
    residual = (y - intercept - x @ phi_prev_arr).copy()
    # lambdas account for the 1/N (vs 1/2N) objective: threshold = alpha*g0*c / 2.
    lambdas = 0.5 * alpha * g0_scale * costs
    column_norms = np.mean(x * x, axis=0)

    objective_history = [
        _transaction_cost_lasso_components(y, x, phi_prev_arr, phi_prev_arr, costs, alpha, g0_scale, intercept)[0]
    ]
    converged = False
    max_coordinate_change = np.inf
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        max_coordinate_change = 0.0

        # Update intercept (unpenalized): optimal given current phi.
        delta_intercept = float(np.mean(residual))
        intercept += delta_intercept
        residual -= delta_intercept
        max_coordinate_change = max(max_coordinate_change, abs(delta_intercept))

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
        objective_value = _transaction_cost_lasso_components(
            y, x, phi, phi_prev_arr, costs, alpha, g0_scale, intercept
        )[0]
        objective_history.append(objective_value)
        n_iter = iteration
        scale = max(1.0, float(np.max(np.abs(phi))), abs(intercept))
        objective_change = abs(objective_history[-2] - objective_history[-1])
        if max_coordinate_change <= tol * scale or objective_change <= tol * scale:
            converged = True
            break

    phi = phi_prev_arr + trade
    objective_value, fit_loss, transaction_penalty = _transaction_cost_lasso_components(
        y, x, phi, phi_prev_arr, costs, alpha, g0_scale, intercept
    )
    return TransactionCostLassoResult(
        phi=phi,
        trade=trade.copy(),
        alpha=alpha,
        intercept=intercept,
        g0_scale=g0_scale,
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
    g0_scale: float = 1.0,
    alpha_grid: np.ndarray | Iterable[float] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-10,
    return_details: bool = False,
):
    """Select alpha by validation AIC over independent validation scenarios.

    Paper §4.2 eq.(24): AIC(alpha) = M*log(RSS/M) + 2*(1 + active_instruments).
    RSS is computed on M independent validation scenarios using the A_t and
    phi_t fitted on the training scenarios. The validation intercept is not
    re-estimated. The +1 counts the unpenalized intercept A_t as a free
    parameter, and active_instruments counts nonzero fitted positions.
    """

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

    m_val = y_val.shape[0]
    rows = []
    best_row = None
    for alpha in alpha_values:
        result = solve_transaction_cost_lasso(
            target_changes,
            hedge_changes,
            phi_prev_arr,
            c_i,
            float(alpha),
            g0_scale=g0_scale,
            max_iter=max_iter,
            tol=tol,
        )
        validation_residual = y_val - result.intercept - x_val @ result.phi
        validation_mse = float(np.mean(validation_residual * validation_residual))
        active_positions = int(np.count_nonzero(np.abs(result.phi) > 1e-8))
        # +1 counts the unpenalized intercept A_t (paper eq.24).
        aic = m_val * np.log(max(validation_mse, np.finfo(float).tiny)) + 2.0 * (1 + active_positions)
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
    true_intercept = 0.2
    # Need n > p+1 (overdetermined) for intercept + phi to be jointly identifiable.
    # Use n=8 scenarios with a random full-rank x; no noise so exact recovery expected.
    rng = np.random.default_rng(0)
    x_full_rank = rng.standard_normal((8, 3))
    y_full_rank = true_intercept + x_full_rank @ phi_true
    x_aug = np.column_stack([np.ones(8), x_full_rank])
    lstsq_sol = np.linalg.lstsq(x_aug, y_full_rank, rcond=None)[0]
    ols_result = solve_transaction_cost_lasso(
        y_full_rank,
        x_full_rank,
        phi_prev,
        np.zeros(3),
        alpha=0.0,
        g0_scale=1.0,
        max_iter=500,
        tol=1e-12,
    )
    if not np.allclose(ols_result.phi, lstsq_sol[1:], atol=1e-8):
        failures.append("alpha=0 with zero costs did not match deterministic OLS phi")
    if not np.isclose(ols_result.intercept, lstsq_sol[0], atol=1e-8):
        failures.append("alpha=0 with zero costs did not match deterministic OLS intercept")

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
    shrink_y = true_intercept + shrink_x @ phi_true
    low_alpha = solve_transaction_cost_lasso(
        shrink_y, shrink_x, phi_prev, np.ones(3), alpha=0.01, g0_scale=1.0
    )
    high_alpha = solve_transaction_cost_lasso(
        shrink_y, shrink_x, phi_prev, np.ones(3), alpha=0.5, g0_scale=1.0
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
        g0_scale=1.0,
        max_iter=500,
    )
    if abs(cost_result.trade[1]) >= abs(cost_result.trade[0]):
        failures.append("higher-cost correlated instrument was not penalized more than cheaper instrument")

    history = np.asarray(low_alpha.objective_history, dtype=float)
    if not np.all(np.isfinite(history)):
        failures.append("objective history contains non-finite values")
    if np.any(np.diff(history) > 1e-10):
        failures.append("objective history increased during coordinate descent")

    train_x = shrink_x
    train_y = true_intercept + train_x @ phi_true + np.array([0.02, -0.01, 0.01, -0.02, 0.0, 0.015])
    validation_x = np.array(
        [
            [0.8, 0.1, 0.2],
            [0.1, 0.9, -0.2],
            [-0.2, 0.3, 0.7],
            [1.1, -0.1, 0.1],
        ]
    )
    validation_y = true_intercept + validation_x @ phi_true + np.array([0.01, -0.015, 0.005, -0.01])
    grid = np.array([0.10, 0.05, 0.01])
    manual_rows = []
    for alpha in grid:
        result = solve_transaction_cost_lasso(
            train_y,
            train_x,
            phi_prev,
            np.ones(3),
            float(alpha),
            g0_scale=1.0,
        )
        validation_residual = validation_y - result.intercept - validation_x @ result.phi
        validation_mse = float(np.mean(validation_residual * validation_residual))
        active_positions = int(np.count_nonzero(np.abs(result.phi) > 1e-8))
        m = validation_y.shape[0]
        aic = m * np.log(max(validation_mse, np.finfo(float).tiny)) + 2.0 * (1 + active_positions)
        manual_rows.append(
            {
                "alpha": float(alpha),
                "aic": float(aic),
                "validation_mse": validation_mse,
                "active_positions": active_positions,
                "intercept": result.intercept,
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
        phi_prev,
        np.ones(3),
        g0_scale=1.0,
        alpha_grid=grid,
    )
    if not np.isclose(selected_alpha, expected_alpha):
        failures.append(
            f"AIC selector selected {selected_alpha}, expected {expected_alpha}; rows={manual_rows}"
        )

    return failures


@dataclass(frozen=True)
class BenchmarkHedgePositions:
    """Greek-matching benchmark hedge positions for one rebalance date."""

    delta: np.ndarray
    delta_vega: np.ndarray


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
    # Paper §2 eq.(6): Z_t = V_t - Pi_t (cumulative from window start, Z_0=0).
    cumulative_tracking_error: float


@dataclass(frozen=True)
class BacktestSummary:
    """Daily backtest outputs plus intervals skipped for incomplete quotes."""

    results: pd.DataFrame
    skipped_intervals: pd.DataFrame
    skipped_interval_count: int


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


def _optionid_equals(value: object, expected: object) -> bool:
    if value == expected:
        return True
    if str(value) == str(expected):
        return True
    try:
        return float(value) == float(expected)
    except (TypeError, ValueError):
        return False


def _single_underlying_index(current_hedges: pd.DataFrame) -> int:
    indices = np.flatnonzero(_is_underlying_frame(current_hedges).to_numpy(dtype=bool))
    if indices.size != 1:
        raise ValueError(f"paper benchmark requires exactly one underlying hedge row; found {indices.size}")
    return int(indices[0])


def _single_atm_index(current_hedges: pd.DataFrame, atm_optionid: object) -> int:
    if atm_optionid is None:
        raise ValueError("paper delta-vega benchmark requires a fixed inception-ATM option id")
    matches = current_hedges["optionid"].map(lambda value: _optionid_equals(value, atm_optionid)).to_numpy(dtype=bool)
    matches = matches & (~_is_underlying_frame(current_hedges).to_numpy(dtype=bool))
    indices = np.flatnonzero(matches)
    if indices.size != 1:
        raise ValueError(f"fixed inception-ATM option {atm_optionid!r} not found exactly once in current hedges")
    return int(indices[0])


def benchmark_hedge_positions(
    current_target: pd.DataFrame,
    current_hedges: pd.DataFrame,
    atm_optionid: object = None,
) -> BenchmarkHedgePositions:
    """Return paper benchmark positions.

    Delta uses only the underlying. Delta-vega uses only the fixed
    inception-ATM option and the underlying. No least-squares fallback is
    allowed for strategies carrying the paper benchmark names.
    """

    target_delta = float(pd.to_numeric(current_target['delta'], errors='coerce').sum())
    target_vega = float(pd.to_numeric(current_target['vega'], errors='coerce').sum())
    hedge_delta = pd.to_numeric(current_hedges['delta'], errors='coerce').to_numpy(dtype=float)
    hedge_vega = pd.to_numeric(current_hedges['vega'], errors='coerce').to_numpy(dtype=float)

    if len(current_hedges) == 0:
        raise ValueError("paper benchmarks require at least one hedge instrument")
    if not np.all(np.isfinite(hedge_delta)) or not np.all(np.isfinite(hedge_vega)):
        raise ValueError("paper benchmark Greeks must be finite")

    underlying_idx = _single_underlying_index(current_hedges)
    underlying_delta = float(hedge_delta[underlying_idx])
    if abs(underlying_delta) <= np.finfo(float).eps:
        raise ValueError("underlying hedge row must have nonzero delta")

    delta_positions = np.zeros(len(current_hedges), dtype=float)
    delta_positions[underlying_idx] = target_delta / underlying_delta

    atm_idx = _single_atm_index(current_hedges, atm_optionid)
    atm_vega = float(hedge_vega[atm_idx])
    if abs(atm_vega) <= np.finfo(float).eps:
        raise ValueError("fixed inception-ATM option has negligible vega")
    phi_atm = target_vega / atm_vega
    phi_underlying = (target_delta - phi_atm * float(hedge_delta[atm_idx])) / underlying_delta
    delta_vega_positions = np.zeros(len(current_hedges), dtype=float)
    delta_vega_positions[atm_idx] = phi_atm
    delta_vega_positions[underlying_idx] = phi_underlying

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
        'cumulative_tracking_error': result.cumulative_tracking_error,
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
    prev_cumulative_z: float = 0.0,
) -> DailyBacktestResult:
    trade = positions - previous_positions
    hedge_change = float(hedge_changes @ positions)
    hedge_delta_exposure = float(hedge_delta @ positions)
    hedge_vega_exposure = float(hedge_vega @ positions)
    transaction_cost = float(np.sum(half_spreads * np.abs(trade)))
    before_cost = float(target_change - hedge_change)
    realized_te = float(before_cost + transaction_cost)
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
        realized_tracking_error=realized_te,
        cumulative_tracking_error=prev_cumulative_z + realized_te,
    )


def run_daily_backtest(
    panel: HedgePanel,
    scenario_source: object,
    alpha_grid: np.ndarray | Iterable[float] | None = None,
    strategies: tuple[str, ...] = ('lasso', 'delta', 'delta_vega'),
    fixed_alpha: float | None = None,
    fixed_atm_optionid: object | None = None,
) -> BacktestSummary:
    """Run model-free daily hedging mechanics over adjacent panel dates.

    Realized evaluation uses only observed next-day mids. The sign convention is:
    positions are replicating hedge holdings, so signed realized tracking error is
    target straddle mid-price change minus hedge-portfolio mid-price change minus
    transaction costs from the rebalance trade. At zero rates this is the
    increment of Z = V - Pi, so transaction costs increase Z.

    fixed_alpha: if provided, skip per-day AIC selection and use this alpha for
    all 21 rebalancing days (paper §4.2: alpha selected once at window open).
    The caller is responsible for selecting alpha via select_alpha_aic before
    calling this function with the paper's N=1000 fit + M=100 validation split.
    When None, alpha is re-selected each day from a 70/30 split (legacy mode).
    """

    requested = tuple(str(strategy) for strategy in strategies)
    allowed = {'lasso', 'delta', 'delta_vega'}
    unknown = sorted(set(requested).difference(allowed))
    if unknown:
        raise ValueError(f'unsupported backtest strategies: {unknown}')
    if len(panel.trading_dates) < 2:
        return BacktestSummary(results=pd.DataFrame(), skipped_intervals=pd.DataFrame(), skipped_interval_count=0)
    benchmark_atm_optionid = fixed_atm_optionid if fixed_atm_optionid is not None else panel.fixed_atm_optionid
    if 'delta_vega' in requested and benchmark_atm_optionid is None:
        raise ValueError('delta_vega benchmark requires fixed_atm_optionid or panel.fixed_atm_optionid')

    n_hedges = len(panel.hedges)
    previous_positions = {strategy: np.zeros(n_hedges, dtype=float) for strategy in requested}
    # Z_t = V_t - Pi_t (paper §2 eq.6); accumulated daily within each window.
    cumulative_z: dict[str, float] = {strategy: 0.0 for strategy in requested}
    rows = []
    skipped_frames = []
    # g0 = initial portfolio value, fixed at window open (paper §3 eq.20).
    # Set from the first complete interval's start-date prices; never updated.
    g0_scale: float | None = None

    for start_date, end_date in zip(panel.trading_dates[:-1], panel.trading_dates[1:]):
        current_target, current_hedges, next_target, next_hedges, missing = _complete_interval_quotes(panel, start_date, end_date)
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
        if g0_scale is None:
            g0_scale = float(np.sum(current_target_mid))
            if g0_scale <= 0:
                g0_scale = 1.0

        if 'lasso' in requested:
            scenario_output = _scenario_source_output(scenario_source, panel, start_date, end_date, current_target, current_hedges)
            if fixed_alpha is not None:
                # Paper §4.2: alpha fixed for entire window; solve directly.
                scenarios = adapt_scenarios_to_solver(scenario_output)
                lasso_result = solve_transaction_cost_lasso(
                    scenarios.target_changes,
                    scenarios.hedge_changes,
                    previous_positions['lasso'],
                    half_spreads,
                    float(fixed_alpha),
                    g0_scale=g0_scale,
                )
                alpha_used = float(fixed_alpha)
            else:
                # Legacy: reselect alpha each day using 70/30 split.
                train, validation = _adapt_train_validation_scenarios(scenario_output)
                alpha_used, lasso_result, _ = select_alpha_aic(
                    train.target_changes,
                    train.hedge_changes,
                    validation.target_changes,
                    validation.hedge_changes,
                    previous_positions['lasso'],
                    half_spreads,
                    g0_scale=g0_scale,
                    alpha_grid=alpha_grid,
                    return_details=True,
                )
            daily = _daily_result(start_date, end_date, 'lasso', lasso_result.phi, previous_positions['lasso'], alpha_used, target_change, hedge_changes, target_delta, target_vega, hedge_delta, hedge_vega, half_spreads, cumulative_z['lasso'])
            rows.append(_result_row(daily))
            previous_positions['lasso'] = lasso_result.phi.copy()
            cumulative_z['lasso'] = daily.cumulative_tracking_error

        if 'delta' in requested or 'delta_vega' in requested:
            benchmarks = benchmark_hedge_positions(current_target, current_hedges, atm_optionid=benchmark_atm_optionid)
            for strategy, positions in (('delta', benchmarks.delta), ('delta_vega', benchmarks.delta_vega)):
                if strategy not in requested:
                    continue
                daily = _daily_result(start_date, end_date, strategy, positions, previous_positions[strategy], None, target_change, hedge_changes, target_delta, target_vega, hedge_delta, hedge_vega, half_spreads, cumulative_z[strategy])
                rows.append(_result_row(daily))
                previous_positions[strategy] = positions.copy()
                cumulative_z[strategy] = daily.cumulative_tracking_error

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
    """Return realized tracking-error moments by strategy.

    Columns prefixed 'terminal_' report statistics over the last observed
    cumulative_tracking_error per window (Z_T in paper §2 eq.6). For a single
    one-month panel there is one terminal value per strategy; for the full
    52-window backtest these columns give the paper's Table 2 statistics.
    """

    columns = [
        'strategy',
        'tracking_count',
        'mean_signed_tracking_error',
        'mean_abs_tracking_error',
        'tracking_rmse',
        'tracking_std',
        'before_cost_tracking_rmse',
        'terminal_count',
        'terminal_mean',
        'terminal_std',
        'terminal_var_5pct',
        'terminal_var_2_5pct',
        'terminal_var_1pct',
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

        # Terminal Z_T: last cumulative value per (start_date) window.
        terminal_z = np.array([], dtype=float)
        if 'cumulative_tracking_error' in group.columns and 'end_date' in group.columns:
            g = group.copy()
            g['end_date'] = pd.to_datetime(g['end_date'])
            # Last row per window = row with max end_date per window-start group.
            # For a single-panel backtest the last row is the only terminal value.
            if 'start_date' in g.columns:
                g['start_date'] = pd.to_datetime(g['start_date'])
                last_rows = g.loc[g.groupby('start_date')['end_date'].idxmax()]
            else:
                last_rows = g.iloc[[-1]]
            terminal_z = _finite_array(pd.to_numeric(last_rows['cumulative_tracking_error'], errors='coerce').to_numpy(dtype=float))

        row = {
            'strategy': strategy,
            'tracking_count': int(errors.size),
            'mean_signed_tracking_error': float(np.mean(errors)) if errors.size else float('nan'),
            'mean_abs_tracking_error': float(np.mean(np.abs(errors))) if errors.size else float('nan'),
            'tracking_rmse': _rmse(errors),
            'tracking_std': float(np.std(errors, ddof=0)) if errors.size else float('nan'),
            'before_cost_tracking_rmse': before_cost_rmse,
            'terminal_count': int(terminal_z.size),
            'terminal_mean': float(np.mean(terminal_z)) if terminal_z.size else float('nan'),
            'terminal_std': float(np.std(terminal_z, ddof=0)) if terminal_z.size > 1 else float('nan'),
            'terminal_var_5pct': float(-np.percentile(terminal_z, 5)) if terminal_z.size else float('nan'),
            'terminal_var_2_5pct': float(-np.percentile(terminal_z, 2.5)) if terminal_z.size else float('nan'),
            'terminal_var_1pct': float(-np.percentile(terminal_z, 1)) if terminal_z.size else float('nan'),
        }
        rows.append(row)
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


def run_full_backtest(
    window_start_dates: list[pd.Timestamp],
    m0_list: list[float],
    scenario_source: object,
    data_dir: Path = DATA_DIR,
    n_fit: int = 1000,
    n_val: int = 100,
    alpha_grid: np.ndarray | None = None,
    strategies: tuple[str, ...] = ('lasso', 'delta', 'delta_vega'),
    target_days: int = 30,
) -> pd.DataFrame:
    """Run 52-window paper backtest and return terminal Z_T per window and m0.

    Paper §4.2 protocol:
    - alpha selected once at t=0 per window using n_fit fitting + n_val validation
      scenarios (independent draws from scenario_source).
    - The same alpha is held fixed for all 21 rebalancing days in the window.
    - Terminal Z_T = cumulative tracking error at window expiry.

    The scenario_source must support a two-key dict return for alpha selection:
      {'train': <scenario_type>, 'validation': <scenario_type>}
    or provide ``alpha_scenarios_for_window(panel)`` returning that dict.
    Paper-mode full backtests do not silently fall back to per-day alpha
    selection when this protocol cannot be satisfied.

    Returns a DataFrame with columns:
      window_start, expiry, m0, strategy, terminal_Z_T, skipped_intervals
    One row per (window, m0, strategy).
    """

    rows_out = []
    for window_start in window_start_dates:
        for m0 in m0_list:
            try:
                panel = build_instrument_panel(
                    start_date=window_start,
                    m0=m0,
                    data_dir=data_dir,
                    target_days=target_days,
                )
            except Exception as exc:
                rows_out.append({
                    'window_start': _as_timestamp(window_start),
                    'expiry': pd.NaT,
                    'm0': m0,
                    'strategy': None,
                    'terminal_Z_T': float('nan'),
                    'skipped_intervals': -1,
                    'error': str(exc),
                })
                continue

            # Alpha selection at t=0: draw n_fit + n_val independent scenarios.
            fixed_alphas: dict[str, float | None] = {s: None for s in strategies}
            if 'lasso' in strategies:
                first_date = panel.trading_dates[0]
                second_date = panel.trading_dates[1] if len(panel.trading_dates) > 1 else first_date
                current_target_rows = panel.quotes[
                    (panel.quotes['date'] == first_date) & (panel.quotes['optionid'].isin(panel.target['optionid']))
                ].drop_duplicates('optionid').set_index('optionid').reindex(panel.target['optionid'])
                current_hedge_rows = panel.quotes[
                    (panel.quotes['date'] == first_date) & (panel.quotes['optionid'].isin(panel.hedges['optionid']))
                ].drop_duplicates('optionid').set_index('optionid').reindex(panel.hedges['optionid'])

                try:
                    if hasattr(scenario_source, 'alpha_scenarios_for_window'):
                        alpha_output = scenario_source.alpha_scenarios_for_window(panel)
                    else:
                        alpha_output = _scenario_source_output(
                            scenario_source, panel, first_date, second_date,
                            current_target_rows.reset_index(), current_hedge_rows.reset_index(),
                        )

                    if not isinstance(alpha_output, Mapping) or not (
                        'train' in alpha_output or 'training' in alpha_output
                    ):
                        raise ValueError('paper full backtest requires independent train/validation alpha scenarios')
                    train_arr, val_arr = _adapt_train_validation_scenarios(alpha_output)
                    if train_arr.target_changes.shape[0] != int(n_fit) or val_arr.target_changes.shape[0] != int(n_val):
                        raise ValueError(
                            f'paper alpha selection requires exactly n_fit={n_fit} and n_val={n_val} scenarios; '
                            f'got {train_arr.target_changes.shape[0]} and {val_arr.target_changes.shape[0]}'
                        )

                    half_spreads = pd.to_numeric(
                        current_hedge_rows['half_spread'], errors='coerce'
                    ).fillna(0.0).to_numpy(dtype=float)
                    current_target_mid = pd.to_numeric(
                        current_target_rows['mid_price'], errors='coerce'
                    ).fillna(0.0).to_numpy(dtype=float)
                    g0_scale = float(np.sum(current_target_mid))
                    if g0_scale <= 0:
                        g0_scale = 1.0

                    selected_alpha = select_alpha_aic(
                        train_arr.target_changes,
                        train_arr.hedge_changes,
                        val_arr.target_changes,
                        val_arr.hedge_changes,
                        np.zeros(len(panel.hedges), dtype=float),
                        half_spreads,
                        g0_scale=g0_scale,
                        alpha_grid=alpha_grid,
                    )
                    fixed_alphas['lasso'] = selected_alpha
                except Exception as exc:
                    rows_out.append({
                        'window_start': panel.start_date,
                        'expiry': panel.expiry_date,
                        'm0': m0,
                        'strategy': 'lasso',
                        'terminal_Z_T': float('nan'),
                        'skipped_intervals': -1,
                        'error': f'alpha_selection_failed: {exc}',
                    })
                    continue

            summary = run_daily_backtest(
                panel,
                scenario_source,
                alpha_grid=alpha_grid,
                strategies=strategies,
                fixed_alpha=fixed_alphas.get('lasso'),
                fixed_atm_optionid=panel.fixed_atm_optionid,
            )

            # Extract terminal Z_T (cumulative tracking error at expiry).
            if summary.results.empty:
                for strategy in strategies:
                    rows_out.append({
                        'window_start': panel.start_date,
                        'expiry': panel.expiry_date,
                        'm0': m0,
                        'strategy': strategy,
                        'terminal_Z_T': float('nan'),
                        'skipped_intervals': summary.skipped_interval_count,
                        'error': 'no_results',
                    })
                continue

            for strategy, group in summary.results.groupby('strategy', sort=False):
                group = group.sort_values('end_date')
                terminal_z = float(group.iloc[-1]['cumulative_tracking_error']) if 'cumulative_tracking_error' in group.columns else float('nan')
                rows_out.append({
                    'window_start': panel.start_date,
                    'expiry': panel.expiry_date,
                    'm0': m0,
                    'strategy': strategy,
                    'terminal_Z_T': terminal_z,
                    'skipped_intervals': summary.skipped_interval_count,
                    'error': None,
                })

    return pd.DataFrame(rows_out)


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
    dates = pd.DatetimeIndex(['2020-01-02', '2020-01-03', '2020-01-06'])
    target = pd.DataFrame({'role': ['target', 'target'], 'optionid': [101, 102], 'cp_flag': ['C', 'P'], 'strike': [100.0, 100.0], 'exdate': [pd.Timestamp('2020-02-03'), pd.Timestamp('2020-02-03')]})
    hedges = pd.DataFrame(
        {
            'role': ['hedge', 'hedge', 'hedge', 'underlying'],
            'optionid': [201, 202, 203, UNDERLYING_OPTIONID],
            'cp_flag': ['P', 'C', 'C', UNDERLYING_CP_FLAG],
            'strike': [95.0, 100.0, 105.0, 100.0],
            'exdate': [pd.Timestamp('2020-02-03')] * 4,
        }
    )
    quote_specs = {
        101: ('target', 'C', 100.0, [5.0, 5.4, 5.1], 0.10, [0.52, 0.55, 0.50], [0.20, 0.19, 0.18]),
        102: ('target', 'P', 100.0, [4.8, 4.5, 4.9], 0.10, [-0.48, -0.45, -0.50], [0.21, 0.20, 0.19]),
        201: ('hedge', 'P', 95.0, [2.1, 2.0, 2.2], 0.04, [-0.22, -0.20, -0.24], [0.12, 0.11, 0.10]),
        202: ('hedge', 'C', 100.0, [3.2, 3.5, 3.3], 0.05, [0.50, 0.53, 0.49], [0.18, 0.17, 0.16]),
        203: ('hedge', 'C', 105.0, [1.7, 1.9, 1.8], 0.03, [0.28, 0.30, 0.27], [0.13, 0.12, 0.11]),
        UNDERLYING_OPTIONID: ('underlying', UNDERLYING_CP_FLAG, 100.0, [100.0, 101.0, 102.0], 0.0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
    }
    rows = []
    for optionid, (role, cp_flag, strike, mids, half_spread, deltas, vegas) in quote_specs.items():
        for idx, date in enumerate(dates):
            if optionid == 203 and date == dates[-1]:
                continue
            rows.append({'date': date, 'exdate': pd.Timestamp('2020-02-03'), 'optionid': optionid, 'role': role, 'cp_flag': cp_flag, 'strike': strike, 'mid_price': mids[idx], 'half_spread': half_spread, 'delta': deltas[idx], 'vega': vegas[idx], 'spot': 100.0 + idx, 'days_to_exp': 30 - idx})
    quotes = pd.DataFrame(rows)
    missing = quote_coverage(quotes, pd.concat([target, hedges], ignore_index=True), dates)
    return HedgePanel(
        start_date=dates[0],
        expiry_date=pd.Timestamp('2020-02-03'),
        m0=1.0,
        target=target,
        hedges=hedges,
        quotes=quotes,
        missing_quotes=missing,
        trading_dates=dates,
        fixed_atm_optionid=202,
    )


def backtest_self_check() -> list[str]:
    """Run deterministic checks for daily backtest mechanics."""

    failures = []
    panel = _backtest_self_check_panel()
    summary = run_daily_backtest(panel, DeterministicBacktestScenarioSource(), alpha_grid=np.array([0.0, 0.01, 0.05]))
    if summary.skipped_interval_count != 1:
        failures.append(f'expected one skipped interval, found {summary.skipped_interval_count}')
    if summary.results.shape[0] != 3:
        failures.append(f'expected three strategy rows for one complete interval, found {summary.results.shape[0]}')
    if set(summary.results.get('strategy', [])) != {'lasso', 'delta', 'delta_vega'}:
        failures.append('backtest did not emit all expected strategies')
    if summary.results.empty:
        return failures
    numeric_columns = ['target_change', 'hedge_change', 'target_delta', 'target_vega', 'hedge_delta_exposure', 'hedge_vega_exposure', 'delta_residual', 'vega_residual', 'transaction_cost', 'realized_tracking_error_before_cost', 'realized_tracking_error', 'cumulative_tracking_error']
    for column in numeric_columns:
        if not np.all(np.isfinite(summary.results[column].to_numpy(dtype=float))):
            failures.append(f'{column} contains non-finite values')
    if np.any(summary.results['transaction_cost'].to_numpy(dtype=float) < -1e-12):
        failures.append('transaction costs must be nonnegative')
    first = summary.results.iloc[0]
    expected_net = float(first['target_change']) - float(first['hedge_change']) + float(first['transaction_cost'])
    if not np.isclose(float(first['realized_tracking_error']), expected_net):
        failures.append('realized tracking error does not match paper cost sign convention')
    if 'cumulative_tracking_error' not in summary.results.columns:
        failures.append('cumulative_tracking_error column missing from backtest results')
    else:
        # For a single interval the cumulative Z equals the daily increment.
        if not np.isclose(float(first['cumulative_tracking_error']), float(first['realized_tracking_error'])):
            failures.append('cumulative_tracking_error does not equal realized_tracking_error for the first interval')
    if summary.skipped_intervals.empty or 'missing_quote' not in set(summary.skipped_intervals['reason']):
        failures.append('skipped interval table did not preserve missing quote reason')
    return failures


def paper_output_self_check() -> list[str]:
    """Run deterministic checks for paper-level reporting helpers."""

    failures = []
    fixture_target = pd.DataFrame({'delta': [0.3], 'vega': [0.4]})
    fixture_hedges = pd.DataFrame(
        {
            'role': ['underlying', 'hedge', 'hedge'],
            'optionid': [UNDERLYING_OPTIONID, 'ATM', 'DISTRACTOR'],
            'cp_flag': [UNDERLYING_CP_FLAG, 'C', 'C'],
            'delta': [1.0, 0.4, 0.7],
            'vega': [0.0, 0.2, 0.1],
        }
    )
    fixture_positions = benchmark_hedge_positions(fixture_target, fixture_hedges, atm_optionid='ATM')
    if not np.allclose(fixture_positions.delta, np.array([0.3, 0.0, 0.0])):
        failures.append(f'paper delta benchmark used non-underlying support: {fixture_positions.delta}')
    if not np.allclose(fixture_positions.delta_vega, np.array([-0.5, 2.0, 0.0])):
        failures.append(f'paper delta-vega benchmark used wrong support: {fixture_positions.delta_vega}')

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
        'cumulative_tracking_error',
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
    if np.linalg.matrix_rank(np.vstack([hedge_delta, hedge_vega])) == 2:
        if not np.allclose(delta_vega_rows['delta_residual'].to_numpy(dtype=float), 0.0, atol=1e-10):
            failures.append('delta-vega strategy should have near-zero delta residual in full-rank fixture')
        if not np.allclose(delta_vega_rows['vega_residual'].to_numpy(dtype=float), 0.0, atol=1e-10):
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

    if args.start_date is None or args.m0 is None:
        parser.error(
            "--start-date and --m0 are required unless --solver-self-check, "
            "--scenario-adapter-self-check, --backtest-self-check, or "
            "--paper-output-self-check is set"
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
