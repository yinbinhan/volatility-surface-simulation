"""3-way static-arbitrage CCDF: training data vs base diffusion vs fine-tuned.

Reuses the validated cs_surface + violations math from gen_arbitrage_ccdf.py
(C/S via Black-Scholes, R=0, real grid). Base samples from the pretrained
sample dir; FT samples passed in via --ft-samples (produced by sample_lora.py).
"""
import argparse, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 0.0; DAY = 21
m = np.array([0.6,0.7,0.8,0.9,0.95,1.0,1.05,1.1,1.2,1.3,1.4])
tau = np.array([0.0027397260273972603,0.019230769230769232,0.038461538461538464,
                0.08333333333333333,0.16666666666666666,0.25,0.5,0.75,1.0])
M, TT = np.meshgrid(m, tau)
verf = np.vectorize(math.erf); ncdf = lambda x: 0.5*(1+verf(x/math.sqrt(2)))
dt = tau[1:]-tau[:-1]; dm = m[1:]-m[:-1]; eps = 1e-8

def cs(iv):
    sq = iv*np.sqrt(TT); d1 = (-np.log(M)+(R+0.5*iv**2)*TT)/sq; d2 = d1-sq
    return ncdf(d1)-M*np.exp(-R*TT)*ncdf(d2)

def viol(p):
    diff_t = p[:,:-1,:]-p[:,1:,:]
    l1 = np.maximum(tau[:-1][None,:,None]*diff_t/np.maximum(dt,eps)[None,:,None],0).sum((1,2))
    diff_m = p[:,:,1:]-p[:,:,:-1]; slope = diff_m/np.maximum(dm,eps)[None,None,:]
    l2 = np.maximum(slope,0).sum((1,2)); l3 = np.maximum(slope[:,:,:-1]-slope[:,:,1:],0).sum((1,2))
    return l1, l2, l3

def load_day(arr):
    return viol(cs(np.exp(arr[:,DAY,0])))

def ccdf(x):
    x = np.sort(np.asarray(x).ravel()); return x, 1.0-np.arange(1,x.size+1)/x.size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--ft-samples", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    from pathlib import Path
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    data = np.load(a.data)
    bf = sorted(glob.glob(f"{a.base_dir}/*.npy"),
                key=lambda s: int(s.split("sample_batch")[1].split(".npy")[0]) if "sample_batch" in s else 0)
    base = np.concatenate([np.load(f) for f in bf], 0)
    ft = np.load(a.ft_samples)
    n = min(len(base), len(ft), len(data))
    series = [("training", "#4d4d4d", load_day(data[:n])),
              ("diffusion", "#1f77b4", load_day(base[:n])),
              ("diffusion-FT", "#d62728", load_day(ft[:n]))]

    # summary
    rows = []
    for name, _, (l1, l2, l3) in series:
        rows.append(f"{name},{l1.mean():.6g},{l2.mean():.6g},{l3.mean():.6g},{(l1+l2+l3).mean():.6g}")
    with open(f"{a.out_dir}/arbitrage_3way_summary.csv", "w") as f:
        f.write("model,l1_mean,l2_mean,l3_mean,total_mean\n" + "\n".join(rows) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    for col, (lab, idx) in enumerate([(r"$\ell_1$", 0), (r"$\ell_2$", 1), (r"$\ell_3$", 2)]):
        ax = axes[col]
        for name, color, ls in series:
            arr = ls[idx].ravel(); xs, sv = ccdf(arr); mask = sv > 0
            if mask.sum() == 0: continue
            ax.plot(xs[mask], sv[mask], color=color, lw=2.5, label=name, drawstyle="steps-post")
        ax.set_xscale("symlog", linthresh=1e-7); ax.set_yscale("log")
        ax.set_title(lab, fontsize=24); ax.set_xlabel("violation threshold", fontsize=18)
        if col == 0: ax.set_ylabel("P(violation > x)", fontsize=18)
    axes[-1].legend(loc="lower left", frameon=False, fontsize=16)
    fig.savefig(f"{a.out_dir}/fig_arbitrage_3way_ccdf.png", dpi=200, bbox_inches="tight")
    print("SAVED", a.out_dir, "n=", n)
    print("\n".join(rows))

if __name__ == "__main__":
    main()
