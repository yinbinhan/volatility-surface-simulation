"""Final rerun on the FIXED pipeline (vega-leg freeze + terminal exact-payoff).
All methods, COVID included + excluded, one consistent code path.
Diffusion + FT: n100 (GPU). VolGAN: n1000 (CPU). Then Table 2 + figures."""
import os, time, subprocess, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/yinbinha/volatility-surface-simulation")
PY = "/home/yinbinha/]/envs/adapted/bin/python"
PROC = "/home/yinbinha/VolGAN/data/processed_shared_grid_11x9_surface_20260519_0155_bw"
OM = "/home/yinbinha/VolGAN/data/optionmetrics_spx_20000103_20230228"
TRAIN = "data/diffusion_shared_grid_iv_22_surface_20260519_train_pre20180616/shared_grid_30d_logiv_return.npy"
BASE = "model_results/dfm_shared_grid_30d_logiv_return_ts1782201131_seed20260623/model-epoch-1000.pt"
FOLDED = "results/final_20260711/ft_folded.pt"
VOLGAN = "/home/yinbinha/VolGAN/results/volgan_paper_surface_20260519_0155_bw/paper.pt"
RT = ["--rate-table", "data/implied_rate.csv", "--risk-free-mode", "implied"]
NVAL, MAXW = 100, 60
M0S = ["0.75", "0.8", "0.9", "1.1", "1.2", "1.25"]
GPUS = [1, 2, 4]
OUT = Path("results/locked_20260712"); (OUT/"logs").mkdir(parents=True, exist_ok=True)
STATUS = OUT/"STATUS.txt"

def log(m):
    with open(STATUS,"a") as f: f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")

def dcmd(ckpt, cov, m0, tag, gpu):
    out = OUT/f"{tag}_m{m0}.csv"
    cmd=[PY,"backtest_diffusion.py","--checkpoint",ckpt,"--train-data",TRAIN,"--processed-dir",PROC,
         "--prepared-dir",PROC,"--data-dir",OM,"--m0",m0,"--n-scenarios","100","--n-val",str(NVAL),
         "--max-windows",str(MAXW),"--output",str(out)]+RT
    if cov: cmd.append("--exclude-covid")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=open(OUT/"logs"/f"{tag}_m{m0}.log","w"),stderr=subprocess.STDOUT),out

def vcmd(cov, m0, tag):
    out = OUT/f"{tag}_m{m0}.csv"
    cmd=[PY,"backtest_volgan.py","--checkpoint",VOLGAN,"--prepared-dir",PROC,"--data-dir",OM,"--m0",m0,
         "--n-scenarios","1000","--n-val",str(NVAL),"--max-windows",str(MAXW),"--device","cpu","--output",str(out)]+RT
    if cov: cmd.append("--exclude-covid")
    return subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ),stdout=open(OUT/"logs"/f"{tag}_m{m0}.log","w"),stderr=subprocess.STDOUT),out

def gpu_queue(jobs):
    pend,run,free=list(jobs),{},list(GPUS)
    while pend or run:
        while pend and free:
            ckpt,cov,m0,tag=pend.pop(0); g=free.pop(0)
            p,o=dcmd(ckpt,cov,m0,tag,g); run[g]=(p,f"{tag}_m{m0}",o); log(f"START {tag}_m{m0} GPU{g}")
        time.sleep(30)
        for g,(p,nm,o) in list(run.items()):
            if p.poll() is not None: log(f"DONE {nm} rc={p.returncode} {'ok' if o.exists() else 'MISS'}"); del run[g]; free.append(g)

def cpu_pool(jobs,width=8):
    pend,run=list(jobs),[]
    while pend or run:
        while pend and len(run)<width:
            cov,m0,tag=pend.pop(0); p,o=vcmd(cov,m0,tag); run.append((p,f"{tag}_m{m0}",o)); log(f"START {tag}_m{m0} cpu")
        time.sleep(30)
        for t in list(run):
            p,nm,o=t
            if p.poll() is not None: log(f"DONE {nm} rc={p.returncode} {'ok' if o.exists() else 'MISS'}"); run.remove(t)

def merge(tag):
    fs=sorted(glob.glob(str(OUT/f"{tag}_m*_raw.csv")))
    if not fs: log(f"merge {tag} NONE"); return None
    pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).to_csv(OUT/f"merged_{tag}_raw.csv",index=False)
    log(f"merge {tag} ({len(fs)})"); return str(OUT/f"merged_{tag}_raw.csv")

def rows(raw,cov,mp):
    df=pd.read_csv(raw); out=[]
    for c,l in mp.items():
        if c not in df.columns: continue
        Z=df[c].dropna().values
        if len(Z)==0: continue
        out.append({"covid":cov,"method":l,"n":int(len(Z)),"mean":float(np.mean(Z)),"median":float(np.median(Z)),
            "std":float(np.std(Z)),"var_5pct":float(-np.percentile(Z,5)),"var_2_5pct":float(-np.percentile(Z,2.5)),
            "var_1pct":float(-np.percentile(Z,1))})
    return out

def main():
    log("=== fixed-pipeline rerun start ===")
    vg=os.fork()
    if vg==0:
        cpu_pool([(False,m,"volgan_in") for m in M0S]+[(True,m,"volgan_out") for m in M0S],width=8)
        merge("volgan_in"); merge("volgan_out"); os._exit(0)
    jobs=[(BASE,False,m,"base_in") for m in M0S]+[(BASE,True,m,"base_out") for m in M0S]
    if Path(FOLDED).exists():
        jobs+=[(FOLDED,False,m,"ft_in") for m in M0S]+[(FOLDED,True,m,"ft_out") for m in M0S]
    gpu_queue(jobs)
    for t in ["base_in","base_out","ft_in","ft_out"]: merge(t)
    os.waitpid(vg,0)
    # Table 2
    R=[]
    mp_b={"unhedged":"unhedged","delta":"delta","delta_vega":"delta_vega","diffusion":"diffusion"}
    for cov,tag in [("included","base_in"),("excluded","base_out")]:
        f=OUT/f"merged_{tag}_raw.csv"
        if f.exists(): R+=rows(str(f),cov,mp_b)
    for cov,tag in [("included","volgan_in"),("excluded","volgan_out")]:
        f=OUT/f"merged_{tag}_raw.csv"
        if f.exists(): R+=rows(str(f),cov,{"volgan":"volgan"})
    for cov,tag in [("included","ft_in"),("excluded","ft_out")]:
        f=OUT/f"merged_{tag}_raw.csv"
        if f.exists(): R+=rows(str(f),cov,{"diffusion":"diffusion_ft"})
    if R: pd.DataFrame(R).to_csv(OUT/"TABLE2_fixed.csv",index=False); log("TABLE2_fixed.csv written")
    # figures
    def fig(dr,vr,sub):
        if Path(dr).exists() and Path(vr).exists():
            r=subprocess.run([PY,"plot_hedging_figures.py","--diffusion-raw",str(dr),"--volgan-raw",str(vr),
                "--output-dir",str(OUT/sub)],cwd=ROOT,capture_output=True,text=True); log(f"FIG {sub} rc={r.returncode}")
    fig(OUT/"merged_base_in_raw.csv",  OUT/"merged_volgan_in_raw.csv",  "figs_diffusion_incovid")
    fig(OUT/"merged_base_out_raw.csv", OUT/"merged_volgan_out_raw.csv", "figs_diffusion_excovid")
    fig(OUT/"merged_ft_in_raw.csv",    OUT/"merged_volgan_in_raw.csv",  "figs_ft_incovid")
    fig(OUT/"merged_ft_out_raw.csv",   OUT/"merged_volgan_out_raw.csv", "figs_ft_excovid")
    log("=== fixed-pipeline rerun DONE ===")

if __name__=="__main__": main()
