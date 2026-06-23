"""Evaluate the diffusion hedging backtest into the Cont & Vuletić (2025) tables.

Consumes the per-observation CSV produced by backtest_diffusion.py and renders
Table 2 (pooled across m0, Covid-included and Covid-excluded, with paper and
VolGAN reference rows), Tables 1/3/4, and the figures (pooled and per-m0 Z_t
histograms, diffusion-vs-baseline scatters, time series, and the alpha/straddle
diagnostics). VaR follows the paper sign convention as a positive loss
magnitude: VaR_p = -percentile(Z, p).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Covid window per CLAUDE.md / paper robustness check
COVID_START = pd.Timestamp("2020-02-13")
COVID_END = pd.Timestamp("2020-07-21")

METHOD_COLS = {
    "unhedged": "Z_unhedged",
    "delta": "Z_delta",
    "delta_vega": "Z_delta_vega",
    "diffusion": "Z_diffusion",
}
# Methods shown in the comparison figures (paper compares the three hedges)
FIG_METHODS = ["delta", "delta_vega", "diffusion"]
FIG_LABELS = {"delta": "Delta", "delta_vega": "Delta-vega", "diffusion": "Diffusion"}

# Paper Table 2 reference (Cont & Vuletić Table 8 — pooled, A2/AIC row).
# Std, VaR5%, VaR2.5%, VaR1% (mean/median omitted in source for some rows).
PAPER_REF = {
    "included": {
        "delta":      dict(std=32.70, var_5pct=19.49, var_2_5pct=36.33, var_1pct=58.63),
        "delta_vega": dict(std=29.70, var_5pct=10.90, var_2_5pct=19.70, var_1pct=43.72),
        "diffusion":  dict(std=32.98, var_5pct=12.79, var_2_5pct=23.42, var_1pct=50.79),
    },
    "excluded": {
        "delta":      dict(std=8.40,  var_5pct=13.22, var_2_5pct=22.84, var_1pct=36.66),
        "delta_vega": dict(std=9.34,  var_5pct=9.58,  var_2_5pct=16.67, var_1pct=34.81),
        "diffusion":  dict(std=8.15,  var_5pct=10.55, var_2_5pct=17.32, var_1pct=33.85),
    },
}


def stats(Z) -> dict:
    Z = np.asarray(Z, dtype=float)
    return {
        "n": len(Z),
        "mean": float(np.mean(Z)),
        "median": float(np.median(Z)),
        "std": float(np.std(Z)),
        "var_5pct": float(-np.percentile(Z, 5)),
        "var_2_5pct": float(-np.percentile(Z, 2.5)),
        "var_1pct": float(-np.percentile(Z, 1)),
    }


def _covid_mask(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["date"])
    return (d >= COVID_START) & (d <= COVID_END)


def print_table2(df: pd.DataFrame):
    """Two panels (Covid included / excluded), with paper reference rows."""
    in_covid = _covid_mask(df)
    panels = {"Covid INCLUDED": df, "Covid EXCLUDED": df[~in_covid]}

    for panel_name, sub in panels.items():
        ref_key = "included" if "INCLUDED" in panel_name else "excluded"
        n_obs = len(sub)
        print(f"\n{'='*92}\nTable 2 — {panel_name}  (n={n_obs} observations)")
        header = (f"{'Method':<22} {'Source':<8} {'Mean':>8} {'Median':>8} {'Std':>8} "
                  f"{'VaR5%':>8} {'VaR2.5%':>9} {'VaR1%':>8}")
        print(header)
        print("-" * len(header))
        for method, col in METHOD_COLS.items():
            s = stats(sub[col].values)
            print(f"{FIG_LABELS.get(method, method):<22} {'Ours':<8} "
                  f"{s['mean']:>8.2f} {s['median']:>8.2f} {s['std']:>8.2f} "
                  f"{s['var_5pct']:>8.2f} {s['var_2_5pct']:>9.2f} {s['var_1pct']:>8.2f}")
            ref = PAPER_REF[ref_key].get(method)
            if ref:
                print(f"{FIG_LABELS.get(method, method):<22} {'Paper':<8} "
                      f"{'—':>8} {'—':>8} {ref['std']:>8.2f} "
                      f"{ref['var_5pct']:>8.2f} {ref['var_2_5pct']:>9.2f} {ref['var_1pct']:>8.2f}")


def fig6_pooled_hist(df: pd.DataFrame, figdir: Path):
    """Pooled histogram of Z_t per hedge method (log-density)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(df[METHOD_COLS[m]].quantile(0.005) for m in FIG_METHODS)
    hi = max(df[METHOD_COLS[m]].quantile(0.995) for m in FIG_METHODS)
    bins = np.linspace(lo, hi, 120)
    for m in FIG_METHODS:
        ax.hist(df[METHOD_COLS[m]].values, bins=bins, density=True,
                histtype="stepfilled", alpha=0.45, label=FIG_LABELS[m])
    ax.set_yscale("log")
    ax.set_xlabel(r"Tracking error $Z_t$ (USD)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 6 — Distribution of tracking error (pooled over $m_0$)")
    ax.legend(title="Method")
    fig.tight_layout()
    out = figdir / "fig6_pooled_hist.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def _scatter(df: pd.DataFrame, y_method: str, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    m0_vals = sorted(df["m0"].unique())
    cmap = plt.get_cmap("tab10")
    for i, m0 in enumerate(m0_vals):
        s = df[df["m0"] == m0]
        ax.scatter(s["Z_diffusion"], s[METHOD_COLS[y_method]], s=8, alpha=0.5,
                   color=cmap(i % 10), label=f"$m_0$ = {m0:g}")
    # Robust axis limits (1st/99th pct of both series) so a few outliers don't
    # compress the bulk; the x=y line still spans the visible range.
    both = np.concatenate([df["Z_diffusion"].values, df[METHOD_COLS[y_method]].values])
    lo, hi = np.percentile(both, [1, 99])
    pad = 0.05 * (hi - lo)
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, "k-", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Tracking error for diffusion hedging (USD)")
    ax.set_ylabel(f"Tracking error for {FIG_LABELS[y_method].lower()} hedging (USD)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def figs78(df: pd.DataFrame, figdir: Path):
    """Scatter diffusion vs delta (Fig 7) and vs delta-vega (Fig 8), Covid excluded.
    Figs 9/10 = the same with Covid included."""
    ex = df[~_covid_mask(df)]
    _scatter(ex, "delta", "Figure 7 — diffusion vs delta (Covid excluded)",
             figdir / "fig7_scatter_delta_excl_covid.png")
    _scatter(ex, "delta_vega", "Figure 8 — diffusion vs delta-vega (Covid excluded)",
             figdir / "fig8_scatter_deltavega_excl_covid.png")
    _scatter(df, "delta", "Figure 9 — diffusion vs delta (Covid included)",
             figdir / "fig9_scatter_delta_incl_covid.png")
    _scatter(df, "delta_vega", "Figure 10 — diffusion vs delta-vega (Covid included)",
             figdir / "fig10_scatter_deltavega_incl_covid.png")


def fig13_per_m0_hist(df: pd.DataFrame, figdir: Path):
    """Per-m0 histograms of Z_t comparing the three methods (2x3 grid)."""
    m0_vals = sorted(df["m0"].unique())
    ncol = 3
    nrow = int(np.ceil(len(m0_vals) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow), squeeze=False)
    for k, m0 in enumerate(m0_vals):
        ax = axes[k // ncol][k % ncol]
        s = df[df["m0"] == m0]
        lo = min(s[METHOD_COLS[m]].quantile(0.005) for m in FIG_METHODS)
        hi = max(s[METHOD_COLS[m]].quantile(0.995) for m in FIG_METHODS)
        bins = np.linspace(lo, hi, 80)
        for m in FIG_METHODS:
            ax.hist(s[METHOD_COLS[m]].values, bins=bins, density=True,
                    histtype="stepfilled", alpha=0.45, label=FIG_LABELS[m])
        ax.set_yscale("log")
        ax.set_title(f"$m_0$ = {m0:g}")
        ax.set_xlabel(r"$Z_t$ (USD)")
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(len(m0_vals), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Figure 13 — Tracking error distribution per $m_0$")
    fig.tight_layout()
    out = figdir / "fig13_per_m0_hist.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def _panel_grid(m0_vals):
    ncol = 3
    nrow = int(np.ceil(len(m0_vals) / ncol))
    return nrow, ncol


def _timeseries(df: pd.DataFrame, figdir: Path, symlog: bool):
    """Figs 11 (linear) / 12 (symlog): Z_t over time per m0, 3 methods."""
    m0_vals = sorted(df["m0"].unique())
    nrow, ncol = _panel_grid(m0_vals)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True)
    colors = {"delta": "tab:blue", "delta_vega": "tab:orange", "diffusion": "tab:green"}
    for k, m0 in enumerate(m0_vals):
        ax = axes[k // ncol][k % ncol]
        s = df[df["m0"] == m0].sort_values("date")
        for m in FIG_METHODS:
            ax.plot(s["date"], s[METHOD_COLS[m]], lw=0.6, color=colors[m], label=FIG_LABELS[m])
        if symlog:
            ax.set_yscale("symlog", linthresh=50)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(f"$m_0$ = {m0:g}")
        ax.set_ylabel("USD")
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(m0_vals), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    tag = "12 (symlog)" if symlog else "11 (linear)"
    fig.suptitle(f"Figure {tag} — Tracking error $Z_t$ over time per $m_0$")
    fig.tight_layout()
    out = figdir / ("fig12_timeseries_symlog.png" if symlog else "fig11_timeseries.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def fig3_alpha_scatter(df: pd.DataFrame, figdir: Path):
    """Fig 3: AIC-selected alpha per window start, colored by m0.
    alpha is constant within a window; reduce to one point per contiguous block."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, m0 in enumerate(sorted(df["m0"].unique())):
        s = df[df["m0"] == m0].copy()
        # One point per window ≈ per calendar month (α is constant within a window;
        # windows start on the first trading day of each month). Use the month's
        # first observation date and its α.
        s["ym"] = pd.to_datetime(s["date"]).dt.to_period("M")
        per_window = s.sort_values("date").groupby("ym").first().reset_index()
        ax.scatter(per_window["date"], per_window["alpha"], s=16, alpha=0.7,
                   color=cmap(i % 10), label=f"$m_0$ = {m0:g}")
    ax.set_xlabel("Starting date")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"Figure 3 — AIC-selected regularization $\alpha$ per window")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = figdir / "fig3_alpha_per_window.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def fig4_straddle_value(df: pd.DataFrame, figdir: Path):
    """Fig 4: straddle value V_t over time, per m0 (overlaid)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, m0 in enumerate(sorted(df["m0"].unique())):
        s = df[df["m0"] == m0].sort_values("date")
        ax.plot(s["date"], s["V_t"], lw=0.6, color=cmap(i % 10), label=f"$m_0$ = {m0:g}")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.set_title(r"Figure 4 — Straddle value $V_t$ for different $m_0$")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = figdir / "fig4_straddle_value.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def fig5_n_instruments(df: pd.DataFrame, figdir: Path, show_m0=(0.8, 1.1)):
    """Fig 5: number of hedge instruments selected over time (paper: m0=0.8, 1.1)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {0.8: "tab:orange", 1.1: "tab:blue"}
    for m0 in show_m0:
        s = df[np.isclose(df["m0"], m0)].sort_values("date")
        if s.empty:
            continue
        ax.plot(s["date"], s["n_instruments"], drawstyle="steps-mid", lw=0.8,
                color=colors.get(m0), label=f"$m_0$ = {m0:g}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of instruments in the portfolio")
    ax.set_title("Figure 5 — Number of hedge instruments selected")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = figdir / "fig5_n_instruments.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def print_table1(df: pd.DataFrame):
    """Table 1: frequency of each instrument count, per m0."""
    max_n = int(df["n_instruments"].max())
    counts = list(range(0, max_n + 1))
    print(f"\n{'='*60}\nTable 1 — frequency of # instruments selected (per m0)")
    header = f"{'m0':>6} " + " ".join(f"{c:>5}" for c in counts)
    print(header)
    print("-" * len(header))
    for m0 in sorted(df["m0"].unique()):
        s = df[df["m0"] == m0]["n_instruments"]
        row = " ".join(f"{int((s == c).sum()):>5}" for c in counts)
        print(f"{m0:>6.2f} {row}")


def _greek_stats(x):
    x = np.asarray(x, dtype=float)
    return (np.mean(x), np.median(x), np.percentile(x, 95), np.percentile(x, 5), np.std(x))


def tables34_greeks(df: pd.DataFrame):
    """Tables 3 (delta) and 4 (vega): hedged position Z_t vs straddle V_t, per m0."""
    for table_no, gname, zcol, vcol in [
        (3, "Delta", "hedged_delta", "straddle_delta"),
        (4, "Vega",  "hedged_vega",  "straddle_vega"),
    ]:
        print(f"\n{'='*78}\nTable {table_no} — {gname} of hedged position $Z_t$ vs straddle $V_t$")
        print(f"{'':>6} | {'Mean':>18} | {'Median':>18} | {'95%ile':>18} | "
              f"{'5%ile':>18} | {'Std':>18}")
        print(f"{'m0':>6} | {'Z_t':>8} {'V_t':>9} | {'Z_t':>8} {'V_t':>9} | "
              f"{'Z_t':>8} {'V_t':>9} | {'Z_t':>8} {'V_t':>9} | {'Z_t':>8} {'V_t':>9}")
        print("-" * 104)
        for m0 in sorted(df["m0"].unique()):
            s = df[df["m0"] == m0]
            z = _greek_stats(s[zcol]); v = _greek_stats(s[vcol])
            cells = " | ".join(f"{z[j]:>8.3f} {v[j]:>9.3f}" for j in range(5))
            print(f"{m0:>6.2f} | {cells}")


def fig14_alpha_robustness(df: pd.DataFrame, alpha_dir: Path, figdir: Path, m0=0.75):
    """Fig 14: Z_t over time for fixed alphas vs AIC (m0=0.75)."""
    series = {}
    aic = df[np.isclose(df["m0"], m0)].sort_values("date")
    if not aic.empty:
        series["AIC selected"] = aic[["date", "Z_diffusion"]]
    if alpha_dir.exists():
        for f in sorted(alpha_dir.glob("obs_alpha_*.csv")):
            a = f.stem.replace("obs_alpha_", "")
            d = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
            series[f"$\\alpha$ = {a}"] = d[["date", "Z_diffusion"]]
    if len(series) <= 1:
        print("  [skip] fig14 — no alpha-robustness CSVs found in "
              f"{alpha_dir} (run backtest_diffusion.py --m0 0.75 --fixed-alpha ...)")
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, d in series.items():
        ax.plot(d["date"], d["Z_diffusion"], lw=0.7, label=label)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("USD")
    ax.set_title(rf"Figure 14 — Tracking error vs $\alpha$ choice ($m_0$ = {m0:g})")
    ax.legend(fontsize=9, title=r"$\alpha$")
    fig.tight_layout()
    out = figdir / "fig14_alpha_robustness.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


# Excel export

def _table2_df(df: pd.DataFrame, covid_included: bool) -> pd.DataFrame:
    """Table 2 as a DataFrame (one panel): rows = method × source."""
    sub = df if covid_included else df[~_covid_mask(df)]
    ref_key = "included" if covid_included else "excluded"
    rows = []
    for method, col in METHOD_COLS.items():
        s = stats(sub[col].values)
        rows.append({"Method": FIG_LABELS.get(method, method), "Source": "Ours",
                     "N": s["n"], "Mean": s["mean"], "Median": s["median"],
                     "Std": s["std"], "VaR5%": s["var_5pct"],
                     "VaR2.5%": s["var_2_5pct"], "VaR1%": s["var_1pct"]})
        ref = PAPER_REF[ref_key].get(method)
        if ref:
            rows.append({"Method": FIG_LABELS.get(method, method), "Source": "Paper",
                         "N": np.nan, "Mean": np.nan, "Median": np.nan,
                         "Std": ref["std"], "VaR5%": ref["var_5pct"],
                         "VaR2.5%": ref["var_2_5pct"], "VaR1%": ref["var_1pct"]})
    return pd.DataFrame(rows)


def _table1_df(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: per-m0 frequency of each option-instrument count."""
    max_n = int(df["n_instruments"].max())
    rows = []
    for m0 in sorted(df["m0"].unique()):
        s = df[df["m0"] == m0]["n_instruments"]
        row = {"m0": m0}
        row.update({f"#={c}": int((s == c).sum()) for c in range(max_n + 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def _greek_table_df(df: pd.DataFrame, zcol: str, vcol: str) -> pd.DataFrame:
    """Table 3 (delta) / 4 (vega): per-m0 stats of hedged Z_t vs straddle V_t."""
    names = ["Mean", "Median", "95%ile", "5%ile", "Std"]
    rows = []
    for m0 in sorted(df["m0"].unique()):
        s = df[df["m0"] == m0]
        z = _greek_stats(s[zcol]); v = _greek_stats(s[vcol])
        row = {"m0": m0}
        for j, nm in enumerate(names):
            row[f"Zt_{nm}"] = z[j]
            row[f"Vt_{nm}"] = v[j]
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables_excel(df: pd.DataFrame, path: Path):
    """Write Tables 1–4 to a multi-sheet .xlsx workbook."""
    has_diag = "n_instruments" in df.columns
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        _table2_df(df, True).to_excel(xw, sheet_name="Table2_Covid_Included", index=False)
        _table2_df(df, False).to_excel(xw, sheet_name="Table2_Covid_Excluded", index=False)
        if has_diag:
            _table1_df(df).to_excel(xw, sheet_name="Table1_Instruments", index=False)
            _greek_table_df(df, "hedged_delta", "straddle_delta").to_excel(
                xw, sheet_name="Table3_Delta", index=False)
            _greek_table_df(df, "hedged_vega", "straddle_vega").to_excel(
                xw, sheet_name="Table4_Vega", index=False)
    sheets = "Table2 (Covid in/out)" + (" + Tables 1/3/4" if has_diag else "")
    print(f"  saved {path}  [{sheets}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", type=Path, default=Path("results/diffusion_obs.csv"))
    parser.add_argument("--figdir", type=Path, default=Path("results/figures"))
    parser.add_argument("--alpha-dir", type=Path, default=Path("results/alpha_robustness"),
                        help="Dir with obs_alpha_<val>.csv for Fig 14")
    parser.add_argument("--excel", type=Path, default=Path("results/hedging_tables.xlsx"),
                        help="Write Tables 1–4 to this .xlsx workbook")
    args = parser.parse_args()

    df = pd.read_csv(args.obs, parse_dates=["date"])
    print(f"Loaded {len(df)} observations over m0 = {sorted(df['m0'].unique())}")

    print_table2(df)

    has_diag = "n_instruments" in df.columns  # new diagnostic columns present?
    if has_diag:
        print_table1(df)
        tables34_greeks(df)

    if args.excel:
        print(f"\nWriting tables workbook ...")
        write_tables_excel(df, args.excel)

    args.figdir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to {args.figdir} ...")
    # Distribution figures (need only the Z columns)
    fig6_pooled_hist(df, args.figdir)
    figs78(df, args.figdir)
    fig13_per_m0_hist(df, args.figdir)
    _timeseries(df, args.figdir, symlog=False)
    _timeseries(df, args.figdir, symlog=True)
    # Diagnostic figures (need the diagnostic columns)
    if has_diag:
        fig3_alpha_scatter(df, args.figdir)
        fig4_straddle_value(df, args.figdir)
        fig5_n_instruments(df, args.figdir)
        fig14_alpha_robustness(df, args.alpha_dir, args.figdir)
    else:
        print("  [skip] Figs 3/4/5/14 + Tables 1/3/4 — obs CSV lacks diagnostic "
              "columns; re-run backtest_diffusion.py to regenerate.")
    print("Done.")


if __name__ == "__main__":
    main()
