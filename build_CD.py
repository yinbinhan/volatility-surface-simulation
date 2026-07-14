"""Build diffusion-hedging tables and figures from locked raw results.

The reported LASSO methods are base diffusion and fine-tuned diffusion. Classical
unhedged, delta, and delta-vega results are included in the pooled summary.
Per-method diagnostic columns are read from each method's own raw output."""

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
L=Path("results/locked_20260712"); F=Path("results/fixed_20260712")
M0S=[0.75,0.8,0.9,1.1,1.2,1.25]
CMAP={m:c for m,c in zip(M0S,["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"])}
# LASSO methods: label -> (raw tag, Z column)
LASSO={"diffusion":("base","diffusion"),"diffusion-FT":("ft","diffusion")}
def raw(tag,arm="in"): return pd.read_csv(L/f"merged_{tag}_{arm}_raw.csv",parse_dates=["window_start","rebalance_date"])

# ---------- T1: freq of #instruments per m0, per method (COVID included / full) ----------
t1=[]
for lab,(tag,_) in LASSO.items():
    d=raw(tag)
    for m0 in M0S:
        vc=d[np.isclose(d.m0,m0)].n_instruments.value_counts()
        row={"method":lab,"m0":m0}
        for k in range(1,9): row[k]=int(vc.get(k,0))
        t1.append(row)
pd.DataFrame(t1).to_csv(L/"T1_num_instruments_freq.csv",index=False)

# ---------- T3/T4: delta & vega of hedged position (Z_t) + straddle (V_t) per m0 per method ----------
def qstats(x):
    x=np.asarray(x); return dict(mean=np.mean(x),median=np.median(x),q95=np.percentile(x,95),
                                 q5=np.percentile(x,5),std=np.std(x))
t3=[]; t4=[]
for lab,(tag,_) in LASSO.items():
    d=raw(tag)
    for m0 in M0S:
        s=d[np.isclose(d.m0,m0)]
        for tab,hd,st in [(t3,"hedged_delta","straddle_delta"),(t4,"hedged_vega","straddle_vega")]:
            zh=qstats(s[hd]); vt=qstats(s[st])
            tab.append({"method":lab,"m0":m0,
                        "Zt_mean":zh["mean"],"Zt_median":zh["median"],"Zt_q95":zh["q95"],"Zt_q5":zh["q5"],"Zt_std":zh["std"],
                        "Vt_mean":vt["mean"],"Vt_median":vt["median"],"Vt_q95":vt["q95"],"Vt_q5":vt["q5"],"Vt_std":vt["std"]})
pd.DataFrame(t3).to_csv(L/"T3_delta_hedged_per_m0.csv",index=False)
pd.DataFrame(t4).to_csv(L/"T4_vega_hedged_per_m0.csv",index=False)

# ---------- T6: % obs where |Z_t| < 1% of V_t, per method per m0 ----------
t6=[]
for lab,(tag,col) in LASSO.items():
    d=raw(tag)
    for m0 in M0S:
        s=d[np.isclose(d.m0,m0)]
        frac=float((np.abs(s[col]) < 0.01*np.abs(s.V_t)).mean())*100
        t6.append({"method":lab,"m0":m0,"pct_within_1pct":round(frac,2)})
pd.DataFrame(t6).to_csv(L/"T6_within_1pct_per_m0.csv",index=False)

# ---------- re-lock T2 (pooled) + sanity vs fixed ----------
META={"observation","window_id","m0","window_start","rebalance_date","interval_end","row_in_window",
      "n_instruments","V_t","straddle_delta","straddle_vega","hedged_delta","hedged_vega","alpha"}
def poolstats(f,col):
    Z=pd.read_csv(f)[col].dropna().values
    return dict(n=len(Z),mean=np.mean(Z),median=np.median(Z),std=np.std(Z),
               v5=-np.percentile(Z,5),v25=-np.percentile(Z,2.5),v1=-np.percentile(Z,1))
t2=[]
for arm,cov in [("in","included"),("out","excluded")]:
    for col,lab,tag in [("unhedged","unhedged","base"),("delta","delta","base"),("delta_vega","delta_vega","base"),
                        ("diffusion","diffusion","base"),("diffusion","diffusion_ft","ft")]:
        s=poolstats(L/f"merged_{tag}_{arm}_raw.csv",col); s.update(covid=cov,method=lab); t2.append(s)
pd.DataFrame(t2).to_csv(L/"TABLE2_locked.csv",index=False)

# ---------- F3: alpha over time per m0 (per method) ----------
fig,axes=plt.subplots(3,2,figsize=(16,11),sharex=True)
for ax,m0 in zip(axes.ravel(),M0S):
    for lab,(tag,_) in LASSO.items():
        d=raw(tag); s=d[np.isclose(d.m0,m0)].drop_duplicates("window_start").sort_values("window_start")
        ax.plot(s.window_start,s.alpha,marker="o",ms=3,lw=0.8,label=lab)
    ax.set_title(f"$m_0$={m0}",fontsize=11); ax.set_ylabel(r"$\alpha$",fontsize=10); ax.grid(alpha=0.2,ls="--")
axes[0,0].legend(fontsize=8,frameon=False)
fig.suptitle("F3: AIC-selected regularization $\\alpha$ per window, by $m_0$",fontsize=14)
fig.tight_layout(); fig.savefig(L/"F3_alpha_by_m0.png",dpi=140,bbox_inches="tight")

# ---------- F4: straddle value V_t over time per m0 (method-independent, use base) ----------
fig,ax=plt.subplots(figsize=(13,6)); d=raw("base")
for m0 in M0S:
    s=d[np.isclose(d.m0,m0)].sort_values("rebalance_date")
    ax.plot(s.rebalance_date,s.V_t,lw=0.8,color=CMAP[m0],label=f"{m0}")
ax.set_title("F4: Straddle value $V_t$ over time by $m_0$",fontsize=13); ax.set_ylabel("$V_t$ (USD)"); ax.set_xlabel("date")
ax.legend(title="$m_0$",fontsize=9,frameon=False); ax.grid(alpha=0.2,ls="--")
fig.tight_layout(); fig.savefig(L/"F4_straddle_value_by_m0.png",dpi=140,bbox_inches="tight")

# ---------- F5: #instruments over time per m0, per method ----------
fig,axes=plt.subplots(3,2,figsize=(16,11),sharex=True)
for ax,m0 in zip(axes.ravel(),M0S):
    for lab,(tag,_) in LASSO.items():
        d=raw(tag); s=d[np.isclose(d.m0,m0)].sort_values("rebalance_date")
        ax.plot(s.rebalance_date,s.n_instruments,lw=0.7,alpha=0.8,label=lab)
    ax.axvspan(pd.Timestamp("2020-02-13"),pd.Timestamp("2020-07-21"),color="red",alpha=0.06)
    ax.set_title(f"$m_0$={m0}",fontsize=11); ax.set_ylabel("# instr",fontsize=10); ax.grid(alpha=0.2,ls="--")
axes[0,0].legend(fontsize=8,frameon=False)
fig.suptitle("F5: Number of hedging instruments selected over time, by $m_0$ (shaded=COVID)",fontsize=14)
fig.tight_layout(); fig.savefig(L/"F5_ninstruments_by_m0.png",dpi=140,bbox_inches="tight")

print("BUILT T1,T3,T4,T6,TABLE2_locked + F3,F4,F5")
# quick prints
print("\n== T1 (base vs fine-tuned diffusion, m0=0.9 & 1.1) ==")
d1=pd.DataFrame(t1)
print(d1[(d1.m0.isin([0.9,1.1]))][["method","m0",1,2,3,4,5,6,7,8]].to_string(index=False))
print("\n== T6 % within 1% of V_t ==")
print(pd.DataFrame(t6).pivot(index="m0",columns="method",values="pct_within_1pct").to_string())
print("\n== T2 locked vs fixed (std, incl) sanity ==")
lk=pd.DataFrame(t2); 
for _,r in lk[lk.covid=="included"].iterrows(): print(f"  {r.method:<12} std={r['std']:.2f} v1={r['v1']:.2f}")
