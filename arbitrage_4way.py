"""4-way static-arbitrage CCDF: training data / diffusion / diffusion-FT / VolGAN.
VolGAN surfaces generated conditionally on real historical states, pooled."""
import argparse, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import backtest_volgan as bv
from backtest_volgan import build_state_lookup, get_day_state, load_volgan_generator
import volgan_adapter as va

R=0.0; DAY=21
m=np.array([0.6,0.7,0.8,0.9,0.95,1.0,1.05,1.1,1.2,1.3,1.4])
tau=np.array([0.0027397260273972603,0.019230769230769232,0.038461538461538464,
              0.08333333333333333,0.16666666666666666,0.25,0.5,0.75,1.0])
M,TT=np.meshgrid(m,tau); verf=np.vectorize(math.erf); ncdf=lambda x:0.5*(1+verf(x/math.sqrt(2)))
dt=tau[1:]-tau[:-1]; dm=m[1:]-m[:-1]; eps=1e-8
def cs(iv):
    sq=iv*np.sqrt(TT); d1=(-np.log(M)+(R+0.5*iv**2)*TT)/sq; d2=d1-sq
    return ncdf(d1)-M*np.exp(-R*TT)*ncdf(d2)
def viol(p):
    diff_t=p[:,:-1,:]-p[:,1:,:]
    l1=np.maximum(tau[:-1][None,:,None]*diff_t/np.maximum(dt,eps)[None,:,None],0).sum((1,2))
    diff_m=p[:,:,1:]-p[:,:,:-1]; slope=diff_m/np.maximum(dm,eps)[None,None,:]
    l2=np.maximum(slope,0).sum((1,2)); l3=np.maximum(slope[:,:,:-1]-slope[:,:,1:],0).sum((1,2))
    return l1,l2,l3

PROC="/home/yinbinha/VolGAN/data/processed_shared_grid_11x9_surface_20260519_0155_bw"
VG="/home/yinbinha/VolGAN/results/volgan_paper_surface_20260519_0155_bw/paper.pt"
OUT=Path("results/fixed_20260712/arbitrage_4way"); OUT.mkdir(parents=True,exist_ok=True)
DATA="data/diffusion_shared_grid_iv_22_surface_20260519/shared_grid_30d_logiv_return.npy"
BASE="samples/dfm_shared_grid_30d_logiv_return_ts1782201131_seed20260623"
FT="results/fixed_20260712/ft_samples_big.npy"

def volgan_surfaces(n_target=600):
    bv.RATE_TABLE={}
    gen,noise_dim,_,_=load_volgan_generator(Path(VG),"cpu")
    sl=build_state_lookup(PROC); dates,log_iv_rows,closes,log_rets,d2i,mg,tg,go=sl
    N=len(dates); step=max(1,N//n_target); out=[]
    for i in range(30,N,step):
        st=get_day_state(dates[i],d2i,log_iv_rows,closes,log_rets)
        if st is None: continue
        log_iv,spot,rtm1,rtm2,rvol=st
        sp,iv=va.sample_scenarios(gen,log_iv,spot,rtm1,rtm2,rvol,N=1,noise_dim=noise_dim,device="cpu")
        out.append(iv[0])  # [m,tau]=[11,9]
        if len(out)>=n_target: break
    arr=np.stack(out)                    # [n,11,9]
    return np.transpose(arr,(0,2,1))     # -> [n,9,11]=[tau,m]

# ---- gather violations ----
data=np.load(DATA)
bf=sorted(glob.glob(f"{BASE}/*.npy"),key=lambda s:int(s.split("sample_batch")[1].split(".npy")[0]))
base=np.concatenate([np.load(f) for f in bf],0)
ft=np.load(FT)
vg=volgan_surfaces(4600); np.save(OUT/"volgan_surfaces.npy",vg)
nd=len(base)  # training ref matches validated n
series=[("training","#4d4d4d",viol(cs(np.exp(data[:nd,DAY,0])))),
        ("diffusion","#1f77b4",viol(cs(np.exp(base[:,DAY,0])))),
        ("diffusion-FT","#d62728",viol(cs(np.exp(ft[:,DAY,0])))),
        ("VolGAN","#2ca02c",viol(cs(vg)))]
print("counts:",{"training":nd,"diffusion":len(base),"ft":len(ft),"volgan":len(vg)})

rows=["model,l1_mean,l2_mean,l3_mean,total_mean"]
for nm,_,(l1,l2,l3) in series:
    rows.append(f"{nm},{l1.mean():.6g},{l2.mean():.6g},{l3.mean():.6g},{(l1+l2+l3).mean():.6g}")
open(OUT/"arbitrage_4way_summary.csv","w").write("\n".join(rows)+"\n")
def ccdf(x):
    x=np.sort(x.ravel()); return x,1.0-np.arange(1,x.size+1)/x.size
fig,axes=plt.subplots(1,3,figsize=(20,5.2),constrained_layout=True)
for col,(lab,idx) in enumerate([(r"$\ell_1$",0),(r"$\ell_2$",1),(r"$\ell_3$",2)]):
    ax=axes[col]
    for nm,color,ls in series:
        xs,sv=ccdf(ls[idx]); msk=sv>0
        if msk.sum(): ax.plot(xs[msk],sv[msk],color=color,lw=2.4,label=nm,drawstyle="steps-post")
    ax.set_xscale("symlog",linthresh=1e-7); ax.set_yscale("log")
    ax.set_title(lab,fontsize=24); ax.set_xlabel("violation threshold",fontsize=16)
    if col==0: ax.set_ylabel("P(violation > x)",fontsize=16)
axes[-1].legend(loc="lower left",frameon=False,fontsize=15)
fig.savefig(OUT/"fig_arbitrage_4way_ccdf.png",dpi=150,bbox_inches="tight")
print("SAVED",OUT/"fig_arbitrage_4way_ccdf.png")
print("\n".join(rows))
