"""Tier B: per-m0 stat tables (T5 incl, T7 excl) + F12 (symlog Z_t time series per m0)
   + F13 (per-m0 Z_t histograms). All from existing fixed_20260712 raw. No rerun."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.scale import SymmetricalLogScale
from pathlib import Path
OUT=Path("results/fixed_20260712"); M0S=[0.75,0.8,0.9,1.1,1.2,1.25]
METH={"unhedged":("base","unhedged","#000000"),"delta":("base","delta","#7f7f7f"),
      "delta-vega":("base","delta_vega","#ff7f0e"),"VolGAN":("volgan","volgan","#2ca02c"),
      "diffusion":("base","diffusion","#1f77b4"),"diffusion-FT":("ft","diffusion","#d62728")}
def raw(tag,arm): return pd.read_csv(OUT/f"merged_{tag}_{arm}_raw.csv",parse_dates=["rebalance_date"])
def stat(Z):
    return dict(n=len(Z),mean=np.mean(Z),median=np.median(Z),std=np.std(Z),
               var5=-np.percentile(Z,5),var25=-np.percentile(Z,2.5),var1=-np.percentile(Z,1))

# ---- T5 (incl) + T7 (excl): per-m0 stats ----
for arm,tab in [("in","T5_per_m0_covid_included"),("out","T7_per_m0_covid_excluded")]:
    rows=[]
    cache={t:raw(t,arm) for t in set(v[0] for v in METH.values())}
    for m0 in M0S:
        for lab,(tag,col,_) in METH.items():
            d=cache[tag]; z=d[np.isclose(d.m0,m0)][col].dropna().values
            if len(z)==0: continue
            s=stat(z); rows.append(dict(m0=m0,method=lab,**s))
    pd.DataFrame(rows).to_csv(OUT/f"{tab}.csv",index=False)
    print(f"\n=== {tab} (std | VaR1%) ===")
    df=pd.DataFrame(rows)
    for m0 in M0S:
        sub=df[df.m0==m0]; parts=[f"{r.method}={r['std']:.1f}/{r.var1:.1f}" for _,r in sub.iterrows()]
        print(f" m0={m0}: "+"  ".join(parts))

# ---- F12: symlog Z_t time series per m0 (paper Fig 12 style) ----
fig,axes=plt.subplots(3,2,figsize=(17,12),sharex=True)
for ax,m0 in zip(axes.ravel(),M0S):
    for lab,(tag,col,color) in METH.items():
        if lab=="unhedged": continue
        d=raw(tag,"in"); d=d[np.isclose(d.m0,m0)].sort_values("rebalance_date")
        ax.plot(d.rebalance_date,d[col],lw=0.8,color=color,alpha=0.85,label=lab)
    ax.set_yscale("symlog",linthresh=50); ax.axhline(0,color="k",lw=0.4,alpha=0.3)
    ax.axvspan(pd.Timestamp("2020-02-13"),pd.Timestamp("2020-07-21"),color="red",alpha=0.06)
    ax.set_title(f"$m_0$={m0}",fontsize=12); ax.set_ylabel("$Z_t$ (USD)",fontsize=10); ax.grid(alpha=0.2,ls="--")
axes[0,0].legend(fontsize=9,ncol=2,frameon=False)
fig.suptitle("F12: Tracking error $Z_t$ per $m_0$ (symlog, linear in [-50,50]); shaded=COVID",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"F12_tracking_ts_symlog_by_m0.png",dpi=140,bbox_inches="tight")
print("\nSAVED F12")

# ---- F13: per-m0 Z_t histograms (paper style, log count) ----
fig,axes=plt.subplots(3,2,figsize=(17,12))
for ax,m0 in zip(axes.ravel(),M0S):
    data={}
    for lab,(tag,col,color) in METH.items():
        if lab=="unhedged": continue
        d=raw(tag,"in"); data[lab]=(d[np.isclose(d.m0,m0)][col].dropna().values,color)
    allz=np.concatenate([v[0] for v in data.values()]); bins=np.linspace(allz.min(),allz.max(),120)
    for lab,(z,color) in data.items():
        ax.hist(z,bins=bins,color=color,alpha=0.55,label=lab,histtype="stepfilled",edgecolor="none")
    ax.set_yscale("log"); ax.set_ylim(bottom=0.8); ax.set_title(f"$m_0$={m0}",fontsize=12)
    ax.set_xlabel("$Z_t$ (USD)",fontsize=10)
    for s in ("top","right"): ax.spines[s].set_visible(False)
axes[0,0].legend(fontsize=8,frameon=False)
fig.suptitle("F13: Tracking error $Z_t$ histogram per $m_0$ (COVID included, log count)",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"F13_hist_by_m0.png",dpi=140,bbox_inches="tight")
print("SAVED F13")
