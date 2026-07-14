"""Sanity checks for the fixed diffusion-hedging pipeline rerun."""

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("results/fixed_20260712")
OLD = Path("results/final_20260711")
CRASH_MONTHS = ["2020-02", "2020-03", "2020-04"]
TAGS = {
    "base_in": "included",
    "base_out": "excluded",
    "ft_in": "included",
    "ft_out": "excluded",
}

lines: list[str] = []


def report(message: str) -> None:
    lines.append(message)
    print(message)


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["window_start"])


def stats(
    frame: pd.DataFrame, column: str
) -> tuple[int, float, float, float]:
    values = frame[column].dropna().to_numpy(dtype=float)
    if not len(values):
        return 0, 0.0, 0.0, 0.0
    return (
        len(values),
        float(np.std(values)),
        float(-np.percentile(values, 5)),
        float(-np.percentile(values, 1)),
    )


report("=" * 70)
report("SANITY REPORT — fixed diffusion-hedging pipeline")
report("=" * 70)

report("\n[1] WINDOW COUNTS AND CRASH-MONTH COVERAGE")
window_sets: dict[str, set] = {}
for tag, coverage in TAGS.items():
    frame = load(OUT / f"merged_{tag}_raw.csv")
    if frame is None:
        report(f"  {tag:<11} MISSING")
        continue
    months = sorted(frame.window_start.dt.strftime("%Y-%m").unique())
    window_sets[tag] = set(frame.window_start.dt.date.unique())
    present = [
        month
        for month in CRASH_MONTHS
        if any(value.startswith(month) for value in months)
    ]
    report(
        f"  {tag:<11} coverage={coverage:<8} "
        f"windows={frame.window_start.nunique():>3} "
        f"obs={len(frame):>5} crash_present={present}"
    )

report("\n[2] BASE/FINE-TUNED WINDOW-SET ALIGNMENT")
for coverage, group in [
    ("included", ["base_in", "ft_in"]),
    ("excluded", ["base_out", "ft_out"]),
]:
    available = [window_sets[tag] for tag in group if tag in window_sets]
    if len(available) == len(group):
        identical = all(values == available[0] for values in available)
        report(
            f"  {coverage}: identical={identical} "
            f"(sizes {[len(values) for values in available]})"
        )

report("\n[3] COVID-INCLUDED VS COVID-EXCLUDED DESCRIPTIVE STATISTICS")
comparisons = {
    "diffusion": ("base_in", "base_out", "diffusion"),
    "diffusion_ft": ("ft_in", "ft_out", "diffusion"),
    "delta": ("base_in", "base_out", "delta"),
    "delta_vega": ("base_in", "base_out", "delta_vega"),
}
for method, (included_tag, excluded_tag, column) in comparisons.items():
    included = load(OUT / f"merged_{included_tag}_raw.csv")
    excluded = load(OUT / f"merged_{excluded_tag}_raw.csv")
    if (
        included is None
        or excluded is None
        or column not in included.columns
        or column not in excluded.columns
    ):
        continue
    _, std_in, _, var1_in = stats(included, column)
    _, std_out, _, var1_out = stats(excluded, column)
    ratio = std_in / std_out if std_out else 0.0
    report(
        f"  {method:<13} incl std={std_in:6.2f} V1%={var1_in:7.2f} "
        f"| excl std={std_out:6.2f} V1%={var1_out:7.2f} "
        f"| ratio std={ratio:.2f}"
    )

report("\n[4] PRIOR VS FIXED BASE-DIFFUSION RUN")
old_path = OLD / "merged_base_covid_in_raw.csv"
new_path = OUT / "merged_base_in_raw.csv"
if old_path.exists() and new_path.exists():
    old_frame = load(old_path)
    new_frame = load(new_path)
    if old_frame is not None and new_frame is not None:
        _, old_std, _, old_var1 = stats(old_frame, "diffusion")
        _, new_std, _, new_var1 = stats(new_frame, "diffusion")
        report(
            f"  included diffusion OLD std={old_std:6.2f} "
            f"V1={old_var1:7.2f} -> NEW std={new_std:6.2f} "
            f"V1={new_var1:7.2f}"
        )

report("\n[5] HEURISTIC FLAGS")
flags: list[str] = []
for tag in ["base_in", "ft_in"]:
    frame = load(OUT / f"merged_{tag}_raw.csv")
    if frame is not None and frame.window_start.nunique() < 48:
        flags.append(
            f"{tag} only {frame.window_start.nunique()} windows (<48)"
        )

base_in = load(OUT / "merged_base_in_raw.csv")
base_out = load(OUT / "merged_base_out_raw.csv")
if base_in is not None and base_out is not None:
    _, std_in, _, _ = stats(base_in, "delta")
    _, std_out, _, _ = stats(base_out, "delta")
    ratio = std_in / std_out if std_out else 0.0
    if ratio < 1.3:
        flags.append(
            f"delta incl/excl std ratio {ratio:.2f} "
            "(<1.3; inspect COVID coverage)"
        )

report("  FLAGS: " + ("; ".join(flags) if flags else "none"))
report_path = OUT / "SANITY_REPORT.txt"
report_path.write_text("\n".join(lines) + "\n")
print("\nwrote", report_path)
