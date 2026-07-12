"""Tracking-error scatter colored by m0 (paper Fig 7-10 style).
x = data-driven (diffusion) Z_t, y = benchmark Z_t, one color per m0, y=x line."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("results/fixed_20260712")
M0S=[0.75,0.8,0.9,1.1,1.2,1.25]
CMAP={m:c for m,c in zip(M0S, ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"])}
def raw(tag,arm): return pd.read_csv(OUT/f"merged_{tag}_{arm}_raw.csv")

def scatter(ax, x, y, m0col, xlab, ylab, title):
    for m0 in M0S:
        mask=np.isclose(m0col,m0)
        ax.scatter(x[mask],y[mask],s=6,alpha=0.5,color=CMAP[m0],label=f"{m0}",edgecolors="none")
    lo=float(min(np.nanmin(x),np.nanmin(y))); hi=float(max(np.nanmax(x),np.nanmax(y)))
    ax.plot([lo,hi],[lo,hi],"k--",lw=1,alpha=0.7)
    ax.set_xlabel(xlab,fontsize=10); ax.set_ylabel(ylab,fontsize=10); ax.set_title(title,fontsize=11)
    ax.grid(alpha=0.2,ls="--")

# diffusion (x) vs delta / delta-vega (y), COVID excl + incl  (= Fig 7,8,9,10)
fig,axes=plt.subplots(2,2,figsize=(14,12))
combos=[("delta","out","F7: delta vs diffusion (COVID excl)"),
        ("delta_vega","out","F8: delta-vega vs diffusion (COVID excl)"),
        ("delta","in","F9: delta vs diffusion (COVID incl)"),
        ("delta_vega","in","F10: delta-vega vs diffusion (COVID incl)")]
for ax,(bench,arm,title) in zip(axes.ravel(),combos):
    d=raw("base",arm)
    scatter(ax,d["diffusion"].values,d[bench].values,d["m0"].values,
            "diffusion $Z_t$ (USD)", f"{bench} $Z_t$ (USD)", title)
axes[0,0].legend(title="$m_0$",fontsize=8,markerscale=2,frameon=False)
fig.suptitle("Tracking-error scatter, colored by $m_0$ (dashed = y=x)",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"F7-10_scatter_by_m0_diffusion.png",dpi=140,bbox_inches="tight")
print("SAVED F7-10 diffusion")

# extra: diffusion (x) vs VolGAN (y), colored by m0, both arms (our head-to-head)
fig,axes=plt.subplots(1,2,figsize=(15,6.5))
for ax,(arm,title) in zip(axes,[("out","diffusion vs VolGAN (COVID excl)"),("in","diffusion vs VolGAN (COVID incl)")]):
    b=raw("base",arm); v=raw("volgan",arm)
    key=["window_id","m0","rebalance_date","row_in_window"]
    mg=b[key+["diffusion"]].merge(v[key+["volgan"]],on=key,how="inner")
    scatter(ax,mg["diffusion"].values,mg["volgan"].values,mg["m0"].values,
            "diffusion $Z_t$","VolGAN $Z_t$",title)
axes[0].legend(title="$m_0$",fontsize=8,markerscale=2,frameon=False)
fig.suptitle("diffusion vs VolGAN tracking error, colored by $m_0$",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"scatter_diffusion_vs_volgan_by_m0.png",dpi=140,bbox_inches="tight")
print("SAVED diffusion-vs-volgan")
