#!/usr/bin/env python3
"""Build shared 11x9 OptionMetrics surfaces for diffusion experiments."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MONEYNESS_GRID = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4], dtype=float)
TAU_GRID = np.array([1 / 252, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1.0], dtype=float)
GRID_ORDER = "m_major_tau_minor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/optionmetrics_spx_20000103_20230228")
    parser.add_argument("--output-dir", default="data/processed_shared_grid_11x9")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--h1", type=float, default=0.01)
    parser.add_argument("--h2", type=float, default=0.01)
    parser.add_argument("--min-otm-quotes", type=int, default=20)
    parser.add_argument("--min-call-quotes", type=int, default=10)
    parser.add_argument("--min-put-quotes", type=int, default=10)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def detect_close_column(df: pd.DataFrame) -> str:
    lower_to_original = {col.lower(): col for col in df.columns}
    for name in ["close", "spx_close", "adj_close", "price", "prc", "last"]:
        if name in lower_to_original:
            return lower_to_original[name]
    raise ValueError(f"could not detect underlying close column from {list(df.columns)}")


def load_underlying(data_root: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    files = sorted((data_root / "underlying").glob("spx_secprd_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no underlying files found under {data_root / 'underlying'}")
    frames = []
    close_col = None
    for path in files:
        df = pd.read_csv(path)
        if close_col is None:
            close_col = detect_close_column(df)
        frames.append(df[["date", close_col]].rename(columns={close_col: "spx_close"}))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["spx_close"] = pd.to_numeric(out["spx_close"], errors="coerce")
    out = out[np.isfinite(out["spx_close"]) & (out["spx_close"] > 0)]
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["spx_prev_close"] = out["spx_close"].shift(1)
    out["log_return"] = np.log(out["spx_close"] / out["spx_prev_close"])
    out.loc[out["spx_prev_close"].isna(), "log_return"] = np.nan
    out["sqrt252_log_return"] = np.sqrt(252.0) * out["log_return"]
    out["trading_day_index"] = np.arange(len(out), dtype=int)
    if start_date:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    out = out.reset_index(drop=True)
    return out


def option_file(data_root: Path, year: int) -> Path:
    path = data_root / "raw_options" / f"spx_options_{year}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def normalized_options(df: pd.DataFrame, close_by_date: dict[pd.Timestamp, float]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    df = df[df["date"].isin(close_by_date)].copy()
    if df.empty:
        return df
    df["spot"] = df["date"].map(close_by_date).astype(float)
    if "days_to_exp" not in df.columns:
        df["days_to_exp"] = (df["exdate"] - df["date"]).dt.days
    if "ttm" not in df.columns:
        df["ttm"] = df["days_to_exp"] / 365.0
    if "strike" not in df.columns:
        if "strike_price" not in df.columns:
            raise ValueError("raw option file needs strike or strike_price")
        df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce") / 1000.0
    if "moneyness" not in df.columns:
        df["moneyness"] = df["strike"] / df["spot"]
    if "mid_price" not in df.columns:
        df["mid_price"] = 0.5 * (df["best_bid"] + df["best_offer"])
    if "half_spread" not in df.columns:
        df["half_spread"] = 0.5 * (df["best_offer"] - df["best_bid"])
    for col in ["spot", "days_to_exp", "ttm", "strike", "moneyness", "best_bid", "best_offer", "mid_price", "half_spread", "impl_volatility"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "vega" in df.columns:
        df["vega"] = pd.to_numeric(df["vega"], errors="coerce")
    valid = (
        (df["exdate"] > df["date"])
        & (df["days_to_exp"] > 0)
        & (df["ttm"] > 0)
        & np.isfinite(df["spot"])
        & np.isfinite(df["moneyness"])
        & df["moneyness"].between(0.5, 1.5)
        & np.isfinite(df["best_bid"])
        & np.isfinite(df["best_offer"])
        & (df["best_bid"] >= 0)
        & (df["best_offer"] >= df["best_bid"])
        & np.isfinite(df["mid_price"])
        & (df["mid_price"] >= 0)
        & np.isfinite(df["half_spread"])
        & (df["half_spread"] >= 0)
        & np.isfinite(df["impl_volatility"])
        & (df["impl_volatility"] > 0)
    )
    if "volume" in df.columns:
        valid &= df["volume"] > 0
    out = df[valid].copy()
    out["cp_flag"] = out["cp_flag"].astype(str).str.upper().str[0]
    return out[out["cp_flag"].isin(["C", "P"])]


def gaussian_smooth(values, m_in, tau_in, h1: float, h2: float, base_weights=None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    m_in = np.asarray(m_in, dtype=float)
    tau_in = np.asarray(tau_in, dtype=float)
    if base_weights is None:
        base_weights = np.ones_like(values, dtype=float)
    else:
        base_weights = np.asarray(base_weights, dtype=float)
        base_weights = np.where(np.isfinite(base_weights) & (base_weights > 0), base_weights, 0.0)
    finite = np.isfinite(values) & np.isfinite(m_in) & np.isfinite(tau_in) & np.isfinite(base_weights)
    values, m_in, tau_in, base_weights = values[finite], m_in[finite], tau_in[finite], base_weights[finite]
    out = np.full((len(MONEYNESS_GRID), len(TAU_GRID)), np.nan)
    local_counts = np.zeros_like(out, dtype=int)
    for i, m0 in enumerate(MONEYNESS_GRID):
        for j, tau0 in enumerate(TAU_GRID):
            kernel = np.exp(-((m_in - m0) ** 2) / (2.0 * h1) - ((tau_in - tau0) ** 2) / (2.0 * h2))
            weights = kernel * base_weights
            denom = weights.sum()
            local_counts[i, j] = int(((np.abs(m_in - m0) <= math.sqrt(h1)) & (np.abs(tau_in - tau0) <= math.sqrt(h2))).sum())
            if denom > 0 and np.isfinite(denom):
                out[i, j] = float(np.sum(values * weights) / denom)
    return out, local_counts


def build_one_date(day: pd.Timestamp, options: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, int], str | None]:
    day_df = options[options["date"] == day]
    calls = day_df[day_df["cp_flag"] == "C"]
    puts = day_df[day_df["cp_flag"] == "P"]
    otm = day_df[((day_df["cp_flag"] == "C") & (day_df["moneyness"] >= 1.0)) | ((day_df["cp_flag"] == "P") & (day_df["moneyness"] < 1.0))]
    counts = {"n_obs_total": int(len(day_df)), "n_otm": int(len(otm)), "n_call": int(len(calls)), "n_put": int(len(puts))}
    if len(otm) < args.min_otm_quotes:
        return {}, counts, "insufficient_otm_quotes"
    if len(calls) < args.min_call_quotes:
        return {}, counts, "insufficient_call_quotes"
    if len(puts) < args.min_put_quotes:
        return {}, counts, "insufficient_put_quotes"
    iv_weights = otm["vega"].to_numpy(float) if "vega" in otm.columns and otm["vega"].notna().any() else None
    iv, iv_local = gaussian_smooth(otm["impl_volatility"], otm["moneyness"], otm["ttm"], args.h1, args.h2, iv_weights)
    call_mid, call_local = gaussian_smooth(calls["mid_price"] / calls["spot"], calls["moneyness"], calls["ttm"], args.h1, args.h2)
    put_mid, put_local = gaussian_smooth(puts["mid_price"] / puts["spot"], puts["moneyness"], puts["ttm"], args.h1, args.h2)
    arrays = {
        "iv": iv,
        "log_iv": np.log(iv),
        "call_mid_over_s": call_mid,
        "put_mid_over_s": put_mid,
        "call_spread_over_s": gaussian_smooth(calls["half_spread"] / calls["spot"], calls["moneyness"], calls["ttm"], args.h1, args.h2)[0],
        "put_spread_over_s": gaussian_smooth(puts["half_spread"] / puts["spot"], puts["moneyness"], puts["ttm"], args.h1, args.h2)[0],
        "call_iv": gaussian_smooth(calls["impl_volatility"], calls["moneyness"], calls["ttm"], args.h1, args.h2)[0],
        "put_iv": gaussian_smooth(puts["impl_volatility"], puts["moneyness"], puts["ttm"], args.h1, args.h2)[0],
        "n_obs_local": iv_local,
        "n_call_obs_local": call_local,
        "n_put_obs_local": put_local,
    }
    for name in ["iv", "log_iv", "call_mid_over_s", "put_mid_over_s", "call_spread_over_s", "put_spread_over_s", "call_iv", "put_iv"]:
        if not np.all(np.isfinite(arrays[name])):
            return arrays, counts, f"nonfinite_{name}"
    if np.any(arrays["iv"] <= 0) or np.any(arrays["call_iv"] <= 0) or np.any(arrays["put_iv"] <= 0):
        return arrays, counts, "nonpositive_iv"
    if np.any(arrays["call_mid_over_s"] < 0) or np.any(arrays["put_mid_over_s"] < 0):
        return arrays, counts, "negative_price"
    if np.any(arrays["call_spread_over_s"] < 0) or np.any(arrays["put_spread_over_s"] < 0):
        return arrays, counts, "negative_spread"
    return arrays, counts, None


def append_surface_rows(date: pd.Timestamp, arrays: dict[str, np.ndarray], counts: dict[str, int], iv_rows: list[dict], price_rows: list[dict]) -> None:
    date_str = date.strftime("%Y-%m-%d")
    for i, m in enumerate(MONEYNESS_GRID):
        for j, tau in enumerate(TAU_GRID):
            iv_rows.append({
                "date": date_str, "moneyness": m, "tau": tau, "tau_days": tau * 365.0,
                "iv": arrays["iv"][i, j], "log_iv": arrays["log_iv"][i, j],
                "n_obs_total": counts["n_obs_total"], "n_obs_local": int(arrays["n_obs_local"][i, j]),
                "smoothing_method": "gaussian_nw_vega_weighted_if_available", "filled_flag": False, "quality_flag": "accepted",
            })
            price_rows.append({
                "date": date_str, "moneyness": m, "tau": tau, "tau_days": tau * 365.0,
                "call_mid_over_s": arrays["call_mid_over_s"][i, j], "put_mid_over_s": arrays["put_mid_over_s"][i, j],
                "call_half_spread_over_s": arrays["call_spread_over_s"][i, j], "put_half_spread_over_s": arrays["put_spread_over_s"][i, j],
                "call_iv": arrays["call_iv"][i, j], "put_iv": arrays["put_iv"][i, j],
                "n_call_obs_local": int(arrays["n_call_obs_local"][i, j]), "n_put_obs_local": int(arrays["n_put_obs_local"][i, j]),
                "filled_flag": False, "quality_flag": "accepted",
            })


def write_outputs(output_dir: Path, spx_daily: pd.DataFrame, accepted_dates: list[pd.Timestamp], tensors: dict[str, list[np.ndarray]], iv_rows: list[dict], price_rows: list[dict], manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_config = {
        "moneyness_grid": MONEYNESS_GRID.tolist(),
        "tau_grid": TAU_GRID.tolist(),
        "tau_days_approx": (TAU_GRID * 365.0).tolist(),
        "grid_order": GRID_ORDER,
        "source_data_dir": manifest["source_data_dir"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_shape_contract": {"surface_points": 99, "surface_return_dim": 100, "conditioning_dim": 102},
    }
    (output_dir / "grid_config.json").write_text(json.dumps(grid_config, indent=2, sort_keys=True) + "\n")
    accepted_daily = spx_daily[spx_daily["date"].isin(accepted_dates)].copy()
    accepted_daily["date"] = accepted_daily["date"].dt.strftime("%Y-%m-%d")
    accepted_daily.to_csv(output_dir / "spx_daily.csv.gz", index=False)
    pd.DataFrame(iv_rows).to_csv(output_dir / "iv_surfaces.csv.gz", index=False)
    pd.DataFrame(price_rows).to_csv(output_dir / "price_surfaces.csv.gz", index=False)
    empty = np.empty((0, len(MONEYNESS_GRID), len(TAU_GRID)))
    np.savez_compressed(
        output_dir / "surface_tensor.npz",
        dates=np.array([d.strftime("%Y-%m-%d") for d in accepted_dates]),
        moneyness_grid=MONEYNESS_GRID,
        tau_grid=TAU_GRID,
        iv=np.stack(tensors["iv"]) if tensors["iv"] else empty,
        log_iv=np.stack(tensors["log_iv"]) if tensors["log_iv"] else empty,
        call_mid_over_s=np.stack(tensors["call_mid_over_s"]) if tensors["call_mid_over_s"] else empty,
        put_mid_over_s=np.stack(tensors["put_mid_over_s"]) if tensors["put_mid_over_s"] else empty,
        call_spread_over_s=np.stack(tensors["call_spread_over_s"]) if tensors["call_spread_over_s"] else empty,
        put_spread_over_s=np.stack(tensors["put_spread_over_s"]) if tensors["put_spread_over_s"] else empty,
        spx_close=accepted_daily["spx_close"].to_numpy(float),
        log_return=accepted_daily["log_return"].to_numpy(float),
    )
    manifest["accepted_dates"] = len(accepted_dates)
    manifest["date_range"] = [accepted_dates[0].strftime("%Y-%m-%d"), accepted_dates[-1].strftime("%Y-%m-%d")] if accepted_dates else []
    manifest["thirty_day_window_count"] = max(0, len(accepted_dates) - 29)
    manifest["file_sizes"] = {path.name: path.stat().st_size for path in sorted(output_dir.iterdir()) if path.is_file()}
    manifest["min_max_stats"] = {
        key: {"min": float(np.nanmin(np.stack(vals))), "max": float(np.nanmax(np.stack(vals))), "nan_count": int(np.isnan(np.stack(vals)).sum())}
        for key, vals in tensors.items() if vals
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    spx_daily = load_underlying(data_root, args.start_date, args.end_date)
    eligible_daily = spx_daily[np.isfinite(spx_daily["log_return"])].copy()
    candidate_daily = eligible_daily.iloc[: args.max_dates].copy() if args.max_dates else eligible_daily.copy()
    close_by_date = dict(zip(candidate_daily["date"], candidate_daily["spx_close"]))
    dates_by_year: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for day in candidate_daily["date"]:
        dates_by_year[int(day.year)].append(day)
    manifest = {
        "source_data_dir": str(data_root),
        "grid": {"moneyness": MONEYNESS_GRID.tolist(), "tau": TAU_GRID.tolist(), "grid_order": GRID_ORDER},
        "smoothing_parameters": {"h1": args.h1, "h2": args.h2, "method": "gaussian_nw"},
        "thresholds": {"min_otm_quotes": args.min_otm_quotes, "min_call_quotes": args.min_call_quotes, "min_put_quotes": args.min_put_quotes},
        "input_files": [],
        "input_row_counts": {},
    }
    accepted_dates: list[pd.Timestamp] = []
    tensors: dict[str, list[np.ndarray]] = defaultdict(list)
    iv_rows: list[dict] = []
    price_rows: list[dict] = []
    drops: Counter[str] = Counter()
    for year in sorted(dates_by_year):
        path = option_file(data_root, year)
        raw = pd.read_csv(path)
        manifest["input_files"].append(str(path))
        manifest["input_row_counts"][str(path)] = int(len(raw))
        opts = normalized_options(raw, close_by_date)
        for day in dates_by_year[year]:
            arrays, counts, drop_reason = build_one_date(day, opts, args)
            if drop_reason:
                drops[drop_reason] += 1
                continue
            accepted_dates.append(day)
            for key in ["iv", "log_iv", "call_mid_over_s", "put_mid_over_s", "call_spread_over_s", "put_spread_over_s"]:
                tensors[key].append(arrays[key])
            append_surface_rows(day, arrays, counts, iv_rows, price_rows)
    manifest["drop_counts_by_reason"] = dict(sorted(drops.items()))
    if not accepted_dates:
        raise RuntimeError(f"no accepted dates; drop counts: {dict(drops)}")
    write_outputs(output_dir, spx_daily, accepted_dates, tensors, iv_rows, price_rows, manifest)
    return output_dir


def self_check(output_dir: Path) -> None:
    text = Path(__file__).read_text()
    forbidden = ("y" + "finance", "pandas_" + "datareader", "yf." + "download", "^" + "GSPC")
    hits = [token for token in forbidden if token in text]
    if hits:
        raise AssertionError(f"external price source references found: {hits}")
    grid = json.loads((output_dir / "grid_config.json").read_text())
    assert grid["moneyness_grid"] == MONEYNESS_GRID.tolist()
    assert np.allclose(grid["tau_grid"], TAU_GRID)
    assert grid["grid_order"] == GRID_ORDER
    daily = pd.read_csv(output_dir / "spx_daily.csv.gz")
    dates = pd.to_datetime(daily["date"])
    assert dates.is_monotonic_increasing and dates.is_unique
    assert np.all(np.isfinite(daily["spx_close"]))
    assert np.all(np.isfinite(daily["log_return"]))
    tensor = np.load(output_dir / "surface_tensor.npz")
    assert tensor["iv"].shape[1:] == (11, 9)
    for key in ["iv", "log_iv", "call_mid_over_s", "put_mid_over_s", "call_spread_over_s", "put_spread_over_s"]:
        arr = tensor[key]
        assert np.all(np.isfinite(arr)), key
    assert np.all(tensor["iv"] > 0)
    assert np.all(tensor["call_mid_over_s"] >= 0)
    assert np.all(tensor["put_mid_over_s"] >= 0)
    assert np.all(tensor["call_spread_over_s"] >= 0)
    assert np.all(tensor["put_spread_over_s"] >= 0)
    iv_long = pd.read_csv(output_dir / "iv_surfaces.csv.gz")
    first_date = str(tensor["dates"][0])
    sample = iv_long[(iv_long["date"] == first_date) & (np.isclose(iv_long["moneyness"], MONEYNESS_GRID[0])) & (np.isclose(iv_long["tau"], TAU_GRID[0]))]
    assert len(sample) == 1
    assert np.isclose(float(sample.iloc[0]["iv"]), float(tensor["iv"][0, 0, 0]))
    manifest = json.loads((output_dir / "audit_manifest.json").read_text())
    assert "thirty_day_window_count" in manifest


def main() -> None:
    args = parse_args()
    output_dir = run(args)
    if args.self_check:
        self_check(output_dir)
        print(f"SELF_CHECK=PASS output_dir={output_dir}")
    else:
        print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
