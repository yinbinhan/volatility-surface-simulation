"""Data-descriptive hedging figures (Cont & Vuletić Figs 1 & 2).

Built from the processed shared-grid market tensor (no model involved): Figure 1
is the average call/put bid-ask spread surfaces (USD), and Figure 2 the average
ATM spread against the arbitrage penalty over the full sample. The per-date
penalty reuses ArbitrageValidator (adapted_sequential_diffusion/fine_tuning.py), which
scores no-arbitrage violations on the call-price (C/S) surface.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from adapted_sequential_diffusion.fine_tuning import ArbitrageValidator


def load_tensor(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    return {
        "dates": pd.to_datetime([str(d) for d in z["dates"].astype(str)]),
        "m": z["moneyness_grid"].astype(float),       # [11]
        "t": z["tau_grid"].astype(float),             # [9]
        "call_spread": z["call_spread_over_s"].astype(np.float32),  # [T,11,9]
        "put_spread": z["put_spread_over_s"].astype(np.float32),    # [T,11,9]
        "call_over_s": z["call_mid_over_s"].astype(np.float32),     # [T,11,9]
        "spx": z["spx_close"].astype(float),          # [T]
    }


def fig1_spread_surfaces(d, figdir: Path, mask):
    """Mean USD bid-ask spread surfaces for calls and puts over the test window."""
    S = d["spx"][mask][:, None, None]                 # [n,1,1]
    call_usd = (d["call_spread"][mask] * S).mean(axis=0)   # [11,9] USD
    put_usd  = (d["put_spread"][mask]  * S).mean(axis=0)   # [11,9]
    M, T = np.meshgrid(d["m"], d["t"], indexing="ij")      # [11,9]

    fig = plt.figure(figsize=(13, 5.5))
    for k, (surf, label) in enumerate([(call_usd, "Calls"), (put_usd, "Puts")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(M, T, surf, cmap="viridis", edgecolor="none")
        ax.set_xlabel("moneyness")
        ax.set_ylabel("time to maturity")
        ax.set_zlabel("bid-ask spread (USD)")
        ax.set_title(f"Mean {label} bid-ask spread")
    fig.suptitle("Figure 1 — Average bid-ask spread for calls and puts")
    fig.tight_layout()
    out = figdir / "fig1_spread_surfaces.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def _arbitrage_penalty_per_date(d) -> np.ndarray:
    """ArbitrageValidator total violation per date on the market C/S surface.

    The validator expects [B,S,C,H,W] with channel 1 = C/S and H=tau, W=moneyness.
    surface_tensor stores [date,moneyness,tau] → transpose to [date,tau,moneyness].
    """
    validator = ArbitrageValidator(d["m"], d["t"])  # grids strictly increasing
    cs = np.transpose(d["call_over_s"], (0, 2, 1))   # [T, 9(tau), 11(moneyness)]
    n = cs.shape[0]
    # [T, S=1, C=2, H=9, W=11]; channel 0 unused by the validator, channel 1 = C/S
    tens = np.zeros((n, 1, 2, cs.shape[1], cs.shape[2]), dtype=np.float32)
    tens[:, 0, 1] = cs
    with torch.no_grad():
        pen = validator.per_sample_total_violation(torch.from_numpy(tens))
    return pen.cpu().numpy()                          # [T]


def fig2_atm_spread_vs_arb(d, figdir: Path):
    """Twin-axis: average ATM bid-ask spread (USD) and arbitrage penalty over time.
    Uses the FULL sample (as in the paper), not just the test window."""
    atm_idx = int(np.argmin(np.abs(d["m"] - 1.0)))   # m=1.0
    atm_spread_usd = d["call_spread"][:, atm_idx, :].mean(axis=1) * d["spx"]  # avg over tau
    penalty = _arbitrage_penalty_per_date(d)

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.scatter(d["dates"], atm_spread_usd, s=3, color="tab:blue", alpha=0.5,
                label="ATM spread")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Average ATM spread (USD)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.scatter(d["dates"], penalty, s=3, color="tab:purple", alpha=0.5,
                label="Arbitrage penalty")
    ax2.set_ylabel("Arbitrage penalty", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    ax1.set_title("Figure 2 — Average ATM bid-ask spread and arbitrage penalty")
    fig.tight_layout()
    out = figdir / "fig2_atm_spread_vs_arbitrage.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-tensor", type=Path,
                        default=Path("data/processed_shared_grid_11x9/surface_tensor.npz"))
    parser.add_argument("--figdir", type=Path, default=Path("results/figures"))
    parser.add_argument("--test-start", default="2018-07-01")
    parser.add_argument("--test-end", default="2023-02-28")
    args = parser.parse_args()

    d = load_tensor(args.surface_tensor)
    mask = np.asarray((d["dates"] >= pd.Timestamp(args.test_start))
                      & (d["dates"] <= pd.Timestamp(args.test_end)))
    print(f"Loaded {len(d['dates'])} dates; {int(mask.sum())} in test window "
          f"{args.test_start}…{args.test_end}")

    args.figdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing data figures to {args.figdir} ...")
    fig1_spread_surfaces(d, args.figdir, mask)
    fig2_atm_spread_vs_arb(d, args.figdir)
    print("Done.")


if __name__ == "__main__":
    main()
