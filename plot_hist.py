"""Z_t histograms in Cont-Vuletic Fig 6 style: filled overlapping bars, log-count y,
linear x full range. 5 methods x {COVID incl, excl}. Outliers included."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("results/fixed_20260712")
SRC={"Delta":("base","delta"),"Delta-vega":("base","delta_vega"),
 "Data-driven (VolGAN)":("volgan","volgan"),"Diffusion":("base","diffusion"),
 "Diffusion-FT":("ft","diffusion")}
COL={"Delta":"#1f77b4","Delta-vega":"#ff7f0e","Data-driven (VolGAN)":"#2ca02c",
 "Diffusion":"#d62728","Diffusion-FT":"#9467bd"}

def get(arm,tag,col): return pd.read_csv(OUT/f"merged_{tag}_{arm}_raw.csv")[col].dropna().values

fig,axes=plt.subplots(1,2,figsize=(16,5.5))
for ax,(arm,title) in zip(axes,[("in","COVID included"),("out","COVID excluded")]):
    data={m:get(arm,t,c) for m,(t,c) in SRC.items()}
    allz=np.concatenate(list(data.values()))
    bins=np.linspace(allz.min(),allz.max(),200)
    for m,z in data.items():   # filled, translucent, overlapping (paper style)
        ax.hist(z,bins=bins,color=COL[m],alpha=0.55,label=m,histtype="stepfilled",edgecolor="none")
    ax.set_yscale("log"); ax.set_ylim(bottom=0.8)
    ax.set_title(f"Tracking error $Z_t$ distribution — {title}",fontsize=13)
    ax.set_xlabel("$Z_t$ (in USD)",fontsize=12); ax.set_ylabel("Density",fontsize=12)
    leg=ax.legend(title="Method",fontsize=10,loc="upper right"); leg.get_title().set_fontsize(10)
    for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
p=OUT/"fig_Zt_hist_paperstyle.png"; fig.savefig(p,dpi=150,bbox_inches="tight"); print("SAVED",p)
