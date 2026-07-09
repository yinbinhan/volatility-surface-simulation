"""Daily net cost-of-carry from put-call parity (for BS pricing off the IV surface).

Per (date, expiry), regress mid(C) - mid(P) on strike K across strikes:
    C - P = e^(-r tau)*(F - K)   =>   slope = -e^(-r tau) = -DF,   intercept = DF*F.
So DF = -slope (true discount) and F = intercept/DF (market forward, dividends included).

For pricing an option off the market IV surface with the code's (spot, r)-Black-Scholes
(which assumes q=0, i.e. forward = spot*e^(r*tau)), the single rate that reproduces the
correct forward is the NET CARRY  b = ln(F/S)/tau = r - q. Using b (not the discount r)
matches the market forward -> correct moneyness/delta/vega, which is what the hedge needs.
The residual discount error (b vs true r) is O(q*tau) ~ 0.1% of price and ignored.

tau uses calendar days / 365 (matches OptionMetrics ACT/365 IV basis; same tau later in BS).

CLI:  python implied_rate.py --data-root <om_root> --start-year 2017 --end-year 2022 \
          --out data/implied_rate.csv --self-check
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MIN_STRIKES = 4
HORIZON_LO_DAYS = 15
HORIZON_HI_DAYS = 120
DF_LO, DF_HI = 0.90, 1.02          # per-expiry discount-factor sanity bounds
CARRY_LO, CARRY_HI = -0.06, 0.08   # per-expiry net-carry sanity bounds


def _detect_close(cols) -> str:
    low = {c.lower(): c for c in cols}
    for name in ["close", "spx_close", "adj_close", "price", "prc", "last"]:
        if name in low:
            return low[name]
    raise ValueError(f"no close column in {list(cols)}")


def load_closes(data_root: Path, start_year: int, end_year: int) -> dict[pd.Timestamp, float]:
    closes: dict[pd.Timestamp, float] = {}
    for year in range(start_year, end_year + 1):
        path = data_root / "underlying" / f"spx_secprd_{year}.csv.gz"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col = _detect_close(df.columns)
        df["date"] = pd.to_datetime(df["date"])
        for d, v in zip(df["date"], pd.to_numeric(df[col], errors="coerce")):
            if np.isfinite(v) and v > 0:
                closes[pd.Timestamp(d)] = float(v)
    return closes


def _prep_options(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    if "strike" not in df.columns:
        df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce") / 1000.0
    if "mid_price" not in df.columns:
        df["mid_price"] = 0.5 * (pd.to_numeric(df["best_bid"], errors="coerce")
                                 + pd.to_numeric(df["best_offer"], errors="coerce"))
    df["cp_flag"] = df["cp_flag"].astype(str).str.upper().str[0]
    keep = (np.isfinite(df["strike"]) & np.isfinite(df["mid_price"])
            & (df["mid_price"] >= 0) & (df["exdate"] > df["date"])
            & df["cp_flag"].isin(["C", "P"]))
    return df[keep][["date", "exdate", "cp_flag", "strike", "mid_price"]]


def _df_forward_one_expiry(calls, puts) -> tuple[float, float] | None:
    """Return (DF, F) from a cross-strike regression of C-P on K, or None."""
    m = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if len(m) < MIN_STRIKES:
        return None
    K = m["strike"].to_numpy(float)
    y = (m["mid_price_c"] - m["mid_price_p"]).to_numpy(float)
    slope, intercept = np.polyfit(K, y, 1)
    resid = np.abs(y - (slope * K + intercept))
    keep = resid <= np.quantile(resid, 0.8)          # one robust trim pass
    if keep.sum() >= MIN_STRIKES:
        slope, intercept = np.polyfit(K[keep], y[keep], 1)
    DF = -slope
    if not (DF_LO <= DF <= DF_HI):
        return None
    return float(DF), float(intercept / DF)


def _carry_one_date(day: pd.DataFrame, spot: float) -> tuple[float, int]:
    date = day["date"].iloc[0]
    carries, near = [], []
    for exdate, g in day.groupby("exdate"):
        days = (exdate - date).days
        tau = days / 365.0
        if tau <= 0:
            continue
        res = _df_forward_one_expiry(g[g.cp_flag == "C"], g[g.cp_flag == "P"])
        if res is None:
            continue
        DF, F = res
        if F <= 0:
            continue
        b = np.log(F / spot) / tau         # net carry r - q
        if not (CARRY_LO <= b <= CARRY_HI):
            continue
        carries.append(b)
        if HORIZON_LO_DAYS <= days <= HORIZON_HI_DAYS:
            near.append(b)
    use = near if near else carries
    return (float(np.median(use)) if use else np.nan, len(use))


def build_rate_table(data_root: Path, start_year: int, end_year: int) -> pd.DataFrame:
    closes = load_closes(data_root, start_year, end_year)
    rows = []
    for year in range(start_year, end_year + 1):
        path = data_root / "raw_options" / f"spx_options_{year}.csv.gz"
        if not path.exists():
            continue
        opt = _prep_options(pd.read_csv(path))
        for date, day in opt.groupby("date"):
            spot = closes.get(pd.Timestamp(date))
            if spot is None:
                continue
            b, n = _carry_one_date(day, spot)
            if np.isfinite(b):
                rows.append({"date": date, "implied_rate": b, "n_expiries": n})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_rate_table(csv_path: Path) -> dict[pd.Timestamp, float]:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return {pd.Timestamp(d): float(r) for d, r in zip(df["date"], df["implied_rate"])}


def rate_lookup(table: dict[pd.Timestamp, float], date: pd.Timestamp, default: float = 0.0) -> float:
    v = table.get(pd.Timestamp(date))
    return float(v) if v is not None and np.isfinite(v) else float(default)


def self_check() -> list[str]:
    """Synthetic chain at known (r,q): regression recovers DF, F; net carry = r-q."""
    fails = []
    r, q, tau, S = 0.024, 0.019, 45 / 365.0, 3000.0
    F = S * np.exp((r - q) * tau)
    DF = np.exp(-r * tau)
    Ks = np.array([2700, 2800, 2900, 3000, 3100, 3200, 3300], float)
    cp = DF * (F - Ks)                       # C - P
    base = np.maximum(F - Ks, 0.0) + 60.0
    calls = pd.DataFrame({"strike": Ks, "mid_price": base})
    puts = pd.DataFrame({"strike": Ks, "mid_price": base - cp})
    res = _df_forward_one_expiry(calls, puts)
    if res is None:
        return ["regression returned None"]
    DF_hat, F_hat = res
    if abs(DF_hat - DF) > 1e-6 or abs(F_hat - F) > 1e-3:
        fails.append(f"DF/F recovered ({DF_hat},{F_hat}) vs ({DF},{F})")
    day = pd.DataFrame({
        "date": pd.Timestamp("2019-06-03"), "exdate": pd.Timestamp("2019-07-18"),
        "cp_flag": ["C"] * len(Ks) + ["P"] * len(Ks),
        "strike": np.r_[Ks, Ks], "mid_price": np.r_[base, base - cp],
    })
    b, n = _carry_one_date(day, S)
    if abs(b - (r - q)) > 1e-6:
        fails.append(f"net carry {b} vs r-q {r-q}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--start-year", type=int, default=2017)
    ap.add_argument("--end-year", type=int, default=2022)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        fails = self_check()
        print("SELF_CHECK=" + ("PASS" if not fails else "FAIL"))
        for f in fails:
            print(" -", f)
        if fails:
            return 1
    if args.data_root and args.out:
        table = build_rate_table(args.data_root, args.start_year, args.end_year)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {len(table)} daily net-carry rates -> {args.out}")
        if len(table):
            print(f"  range {table['date'].min().date()}..{table['date'].max().date()}  "
                  f"b mean={table['implied_rate'].mean():.4f} "
                  f"min={table['implied_rate'].min():.4f} max={table['implied_rate'].max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
