"""Regenerate the four paper hedging figures from the fixed_20260712 pipeline.

Covid-EXCLUDED arm (arm='out', N=6288). AD-Seq = base 'diffusion' config;
VolGAN (Ours) = reproduced 'volgan' run. Style matches the committed figures:
  - fig_te_histogram.png            : log-count histogram, central 99%, "Method" legend
  - fig_diffusion_vs_delta.png      : scatter AD-Seq (x) vs Delta (y), robust 1/99 limits, grid, x=y
  - fig_diffusion_vs_delta_vega.png : scatter AD-Seq (x) vs Delta-vega (y), robust 1/99 limits, grid, x=y
  - fig_diffusion_vs_volgan.png     : scatter VolGAN (Ours) (x) vs AD-Seq (y), central 99%, corr box, x=y
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SRC = Path("results/fixed_20260712")
OUT = SRC / "paper_figs_final"; OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["m0", "window_start", "rebalance_date", "interval_end", "row_in_window"]

base = pd.read_csv(SRC / "merged_base_out_raw.csv")      # delta, delta_vega, diffusion(AD-Seq)
volg = pd.read_csv(SRC / "merged_volgan_out_raw.csv")    # volgan (VolGAN Ours)
merged = base[KEYS + ["delta", "delta_vega", "diffusion"]].merge(
    volg[KEYS + ["volgan"]], on=KEYS)
print("N per arm (Covid excluded):", len(merged))

COLORS = {0.75: "#1f77b4", 0.8: "#ff7f0e", 0.9: "#2ca02c",
          1.1: "#d62728", 1.2: "#9467bd", 1.25: "#8c564b"}

# --------------------------------------------------------------------------
# Figure 1: pooled tracking-error histogram, central 99%, log count
# --------------------------------------------------------------------------
HIST = [("delta", "Delta", "#1f77b4"), ("delta_vega", "Delta-vega", "#ff7f0e"),
        ("volgan", "VolGAN (Ours)", "#2ca02c"), ("diffusion", "AD-Seq", "#d62728")]
allz = np.concatenate([merged[c].to_numpy() for c, _, _ in HIST])
lo, hi = np.quantile(allz, [0.005, 0.995])
bins = np.linspace(lo, hi, 200)
fig, ax = plt.subplots(figsize=(13.5, 7.2))
for col, lab, color in HIST:
    ax.hist(merged[col].to_numpy(), bins=bins, color=color, alpha=0.5,
            label=lab, histtype="bar")
ax.set_yscale("log"); ax.set_ylim(bottom=0.8)
ax.set_title(r"Tracking error $Z_t$ distribution for all values of $m_0$")
ax.set_xlabel(r"$Z_t$ (USD)"); ax.set_ylabel("Count")
ax.legend(title="Method", loc="upper right")
fig.savefig(OUT / "fig_te_histogram.png", dpi=180)
plt.close(fig)

# --------------------------------------------------------------------------
# Figures 2-3: AD-Seq (x) vs Delta / Delta-vega (y), robust 1/99 limits, grid
# --------------------------------------------------------------------------
def scatter_full(y_col, y_label, filename, title):
    x_col = "diffusion"
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    for m0 in sorted(merged["m0"].unique()):
        s = merged[np.isclose(merged["m0"], m0)]
        ax.scatter(s[x_col], s[y_col], s=9, alpha=0.5, linewidths=0,
                   color=COLORS[float(m0)], label=rf"$m_0 = {m0:g}$")
    both = np.concatenate([merged[x_col].to_numpy(), merged[y_col].to_numpy()])
    plo, phi = np.percentile(both, [1, 99]); pad = 0.05 * (phi - plo)
    lims = [plo - pad, phi + pad]
    ax.plot(lims, lims, "k-", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Tracking error for AD-Seq hedging (USD)")
    ax.set_ylabel(f"Tracking error for {y_label} hedging (USD)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=12)
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)

scatter_full("delta", "Delta", "fig_diffusion_vs_delta.png", "AD-Seq vs Delta")
scatter_full("delta_vega", "Delta-vega", "fig_diffusion_vs_delta_vega.png", "AD-Seq vs Delta-vega")

# --------------------------------------------------------------------------
# Figure 4: VolGAN (Ours) (x) vs AD-Seq (y), central 99%, correlation box
# --------------------------------------------------------------------------
x_col, y_col = "volgan", "diffusion"
s = merged[[x_col, y_col, "m0"]].replace([np.inf, -np.inf], np.nan).dropna()
both = np.concatenate([s[x_col].to_numpy(), s[y_col].to_numpy()])
qlo, qhi = np.quantile(both, [0.005, 0.995]); pad = 0.05 * (qhi - qlo)
clo, chi = qlo - pad, qhi + pad
shown = s[(s[x_col] >= clo) & (s[x_col] <= chi) & (s[y_col] >= clo) & (s[y_col] <= chi)]
full_r = s[x_col].corr(s[y_col], method="pearson")
full_rho = s[x_col].corr(s[y_col], method="spearman")
shown_r = shown[x_col].corr(shown[y_col], method="pearson")
shown_rho = shown[x_col].corr(shown[y_col], method="spearman")
fig, ax = plt.subplots(figsize=(6.8, 6.2))
for m0 in sorted(shown["m0"].unique()):
    sub = shown[np.isclose(shown["m0"], m0)]
    ax.scatter(sub[x_col], sub[y_col], s=9, alpha=0.55, linewidths=0,
               color=COLORS[float(m0)], label=rf"$m_0$ = {m0:g}")
ax.plot([clo, chi], [clo, chi], color="black", lw=1.0)
ax.set_xlim(clo, chi); ax.set_ylim(clo, chi)
ax.set_xlabel("Tracking error for VolGAN (Ours) hedging (USD)")
ax.set_ylabel("Tracking error for AD-Seq hedging (USD)")
ax.set_title("VolGAN (Ours) vs AD-Seq tracking error (central 99%)")
ax.text(0.03, 0.97,
        f"shown n={len(shown)} / full n={len(s)}\n"
        f"shown Pearson r={shown_r:.3f}\nshown Spearman rho={shown_rho:.3f}\n"
        f"full Pearson r={full_r:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.85, boxstyle="round,pad=0.3"))
ax.legend(title=r"$m_0$", fontsize=8, loc="lower right", frameon=True)
fig.tight_layout()
fig.savefig(OUT / "fig_diffusion_vs_volgan.png", dpi=160)
plt.close(fig)

print("FULL Pearson(VolGAN, AD-Seq) =", round(float(full_r), 4))
print("SHOWN Pearson(VolGAN, AD-Seq) =", round(float(shown_r), 4),
      "shown n=", len(shown), "full n=", len(s))
print("SAVED to", OUT)
