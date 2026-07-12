"""Time-varying hedging tracking error Z_t per m0 (paper Fig 12/13 analogue), COVID included."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("results/fixed_20260712")
M0S=[0.75,0.8,0.9,1.1,1.2,1.25]
METH={"delta":("base_in","delta","#7f7f7f"),"delta-vega":("base_in","delta_vega","#ff7f0e"),
      "VolGAN":("volgan_in","volgan","#2ca02c"),"diffusion":("base_in","diffusion","#1f77b4"),
      "diffusion-FT":("ft_in","diffusion","#d62728")}
raws={t:pd.read_csv(OUT/f"merged_{t}_raw.csv",parse_dates=["rebalance_date"]) for t in
      set(v[0] for v in METH.values())}
fig,axes=plt.subplots(3,2,figsize=(17,12),sharex=True)
for ax,m0 in zip(axes.ravel(),M0S):
    for lab,(tag,col,color) in METH.items():
        d=raws[tag]; d=d[np.isclose(d.m0,m0)].sort_values("rebalance_date")
        ax.plot(d.rebalance_date,d[col],lw=0.9,color=color,alpha=0.85,label=lab)
    ax.axhline(0,color="k",lw=0.5,alpha=0.3); ax.axvspan(pd.Timestamp("2020-02-13"),pd.Timestamp("2020-07-21"),color="red",alpha=0.06)
    ax.set_title(f"$m_0$={m0}",fontsize=12); ax.set_ylabel("$Z_t$ (USD)",fontsize=10); ax.grid(alpha=0.2,ls="--")
axes[0,0].legend(fontsize=9,ncol=2,frameon=False)
fig.suptitle("Time-varying hedging tracking error $Z_t$ by $m_0$ (COVID included; shaded=COVID window)",fontsize=15)
fig.tight_layout(); p=OUT/"fig_tracking_ts_by_m0.png"; fig.savefig(p,dpi=140,bbox_inches="tight"); print("SAVED",p)
