"""Plot hedging comparison figures from raw tracking-error CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_raw(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"delta", "delta_vega", method}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df


def tracking_stats(values: pd.Series) -> dict[str, float]:
    z = values.dropna().to_numpy(dtype=float)
    return {
        "n": float(len(z)),
        "mean": float(np.mean(z)),
        "median": float(np.median(z)),
        "std": float(np.std(z)),
        "var_5pct": float(-np.percentile(z, 5)),
        "var_2_5pct": float(-np.percentile(z, 2.5)),
        "var_1pct": float(-np.percentile(z, 1)),
    }


def scatter_pair(ax, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    ax.scatter(x, y, s=9, alpha=0.55, linewidths=0)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(min(x[finite].min(), y[finite].min()))
        hi = float(max(x[finite].max(), y[finite].max()))
        pad = 0.05 * max(hi - lo, 1.0)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(f"Tracking error for {x_col} hedging (USD)")
    ax.set_ylabel(f"Tracking error for {y_col} hedging (USD)")
    ax.grid(True, linewidth=0.4, alpha=0.3)


def plot_model(df: pd.DataFrame, method: str, label: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    scatter_pair(axes[0], df, method, "delta", f"{label} vs delta")
    scatter_pair(axes[1], df, method, "delta_vega", f"{label} vs delta-vega")
    out = output_dir / f"{method}_classical_scatter.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_stats(frames: list[tuple[str, pd.DataFrame, str]], output_dir: Path) -> None:
    rows = []
    for model_label, df, method in frames:
        for col in ["unhedged", "delta", "delta_vega", method]:
            if col in df.columns:
                rows.append({"source": model_label, "method": col, **tracking_stats(df[col])})
    pd.DataFrame(rows).to_csv(output_dir / "tracking_error_stats.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volgan-raw", type=Path, default=None)
    parser.add_argument("--diffusion-raw", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[str, pd.DataFrame, str]] = []
    if args.volgan_raw is not None and args.volgan_raw.exists():
        volgan = load_raw(args.volgan_raw, "volgan")
        plot_model(volgan, "volgan", "VolGAN data-driven", args.output_dir)
        frames.append(("volgan", volgan, "volgan"))
    if args.diffusion_raw is not None and args.diffusion_raw.exists():
        diffusion = load_raw(args.diffusion_raw, "diffusion")
        plot_model(diffusion, "diffusion", "Diffusion data-driven", args.output_dir)
        frames.append(("diffusion", diffusion, "diffusion"))
    if not frames:
        raise ValueError("no existing raw CSVs were provided")
    write_stats(frames, args.output_dir)
    print(f"Figures and stats written to {args.output_dir}")


if __name__ == "__main__":
    main()
