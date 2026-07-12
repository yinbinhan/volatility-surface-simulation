"""Sanity checks for the fixed-pipeline rerun. Writes SANITY_REPORT.txt."""
import numpy as np, pandas as pd, glob, sys
from pathlib import Path
OUT = Path("results/fixed_20260712")
OLD = Path("results/final_20260711")
CRASH = ["2020-02","2020-03","2020-04"]
META = {"observation","window_id","m0","window_start","rebalance_date","interval_end","row_in_window"}
lines = []
def p(s): lines.append(s); print(s)

def load(f):
    return pd.read_csv(f, parse_dates=["window_start"]) if Path(f).exists() else None

def stats(df, col):
    Z = df[col].dropna().values
    return (len(Z), np.std(Z), -np.percentile(Z,5), -np.percentile(Z,1)) if len(Z) else (0,0,0,0)

tags = {"base_in":"included","base_out":"excluded","ft_in":"included","ft_out":"excluded",
        "volgan_in":"included","volgan_out":"excluded"}
p("="*70); p("SANITY REPORT — fixed pipeline (vega-freeze + terminal payoff)"); p("="*70)

# 1. window counts + crash presence + fairness
p("\n[1] WINDOW COUNTS (unique window_start) + crash-month presence")
wsets={}
for tag,cov in tags.items():
    df=load(OUT/f"merged_{tag}_raw.csv")
    if df is None: p(f"  {tag:<11} MISSING"); continue
    u=sorted(df.window_start.dt.strftime("%Y-%m").unique()); wsets[tag]=set(df.window_start.dt.date.unique())
    crash=[m for m in CRASH if any(x.startswith(m) for x in u)]
    p(f"  {tag:<11} windows={df.window_start.nunique():>3}  obs={len(df):>5}  crash_present={crash}")

# 2. fairness: same window set across methods within a covid arm
p("\n[2] FAIRNESS (identical window set across methods per covid arm)")
for cov,grp in [("included",["base_in","ft_in","volgan_in"]),("excluded",["base_out","ft_out","volgan_out"])]:
    ss=[wsets[g] for g in grp if g in wsets]
    if len(ss)==len(grp):
        same=all(s==ss[0] for s in ss); p(f"  {cov}: identical={same} (sizes {[len(s) for s in ss]})")

# 3. incl vs excl gap (the COVID spike)
p("\n[3] COVID incl vs excl (std / VaR1%) — expect incl >> excl now")
for meth,(ti,to) in {"diffusion":("base_in","base_out"),"volgan":("volgan_in","volgan_out"),
                     "delta":("base_in","base_out"),"delta_vega":("base_in","base_out"),
                     "diffusion_ft":("ft_in","ft_out")}.items():
    col="diffusion" if meth=="diffusion_ft" else meth
    di,do=load(OUT/f"merged_{ti}_raw.csv"),load(OUT/f"merged_{to}_raw.csv")
    if di is None or do is None or col not in di.columns: continue
    ni,si,_,vi=stats(di,col); no,so,_,vo=stats(do,col)
    p(f"  {meth:<13} incl std={si:6.2f} V1%={vi:7.2f} | excl std={so:6.2f} V1%={vo:7.2f} | ratio std={si/so if so else 0:.2f}")

# 4. old vs new diff
p("\n[4] OLD (broken, results/final_20260711) vs NEW (fixed) — diffusion/volgan")
oldmap={"included":("merged_base_covid_in_raw.csv","merged_volgan_covid_in_raw.csv"),
        "excluded":(None,None)}  # old excl was reused rqtau; compare incl only + note
for cov,(ob,ov) in oldmap.items():
    if ob is None: continue
    for col,of in [("diffusion",ob),("volgan",ov)]:
        oldf=OLD/of; newtag={"diffusion":"base_"+("in" if cov=="included" else "out"),
                             "volgan":"volgan_"+("in" if cov=="included" else "out")}[col]
        newf=OUT/f"merged_{newtag}_raw.csv"
        if oldf.exists() and newf.exists():
            _,os_,_,ov1=stats(load(oldf),col); _,ns_,_,nv1=stats(load(newf),col)
            p(f"  {cov} {col:<10} OLD std={os_:6.2f} V1={ov1:7.2f} -> NEW std={ns_:6.2f} V1={nv1:7.2f}")

# 5. verdict heuristics
p("\n[5] HEURISTIC FLAGS (for confidence judgment)")
flags=[]
for tag in ["base_in","volgan_in"]:
    df=load(OUT/f"merged_{tag}_raw.csv")
    if df is not None and df.window_start.nunique() < 48: flags.append(f"{tag} only {df.window_start.nunique()} windows (<48)")
bi,bo=load(OUT/"merged_base_in_raw.csv"),load(OUT/"merged_base_out_raw.csv")
if bi is not None and bo is not None:
    _,si,_,_=stats(bi,"delta"); _,so,_,_=stats(bo,"delta")
    if si/so < 1.3: flags.append(f"delta incl/excl std ratio {si/so:.2f} (<1.3 -> COVID spike still weak)")
p("  FLAGS: "+("; ".join(flags) if flags else "none — looks healthy"))
open(OUT/"SANITY_REPORT.txt","w").write("\n".join(lines)+"\n")
print("\nwrote", OUT/"SANITY_REPORT.txt")
