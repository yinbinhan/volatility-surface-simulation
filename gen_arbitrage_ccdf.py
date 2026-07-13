"""Figure 3: complementary CDF of static-arbitrage violation magnitude.

Compares four sets of day-22 (next-step target) surfaces, split into the three
no-arbitrage loss components l1 (calendar), l2 (call-spread/vertical), l3 (butterfly):
  * training   -- smoothed market surfaces (real day-22 of each window)
  * VolGAN     -- VolGAN next-day surfaces for the same conditioning windows
  * AD-Seq     -- adaptive sequential diffusion generated day-22 surfaces (base)
  * AD-Seq-ft  -- fine-tuned adaptive sequential diffusion generated day-22 surfaces

All curves use the SAME (m, tau) grid, the SAME C/S Black-Scholes map (R=0), and the
SAME l1/l2/l3 arbitrage formulas, so they are strictly comparable.

Violation arrays for training/AD-Seq/AD-Seq-ft reproduce their reference summary CSVs
to 6 significant figures (day22_violation_summary.csv and fixed_20260712/arbitrage_4way/
arbitrage_4way_summary.csv). VolGAN surfaces are generated on the fly from the trained
surface-mode checkpoint via the same volgan_adapter path used by the hedging ablation.
"""
import sys, glob, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / "VolGAN"))
import VolGAN as _VolGAN  # noqa: E402

R = 0.0
DAY = 21  # target_index (day 22)
LINTHRESH = 1e-7
SEED = 20260623
GEN_DIR = "samples/dfm_shared_grid_30d_logiv_return_ts1782201131_seed20260623"
DATA = "data/diffusion_shared_grid_iv_22_surface_20260519/shared_grid_30d_logiv_return.npy"
FT_SAMPLES = "results/fixed_20260712/ft_samples_big.npy"
VOLGAN_CKPT = str(Path.home() / "VolGAN/results/volgan_paper_surface_20260519_0155_bw/paper.pt")
OUT = "results/paper_protocol_v2/arbitrage_violation_current_ddpm/fig_arbitrage_ccdf.png"

m = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4])
tau = np.array([0.0027397260273972603, 0.019230769230769232, 0.038461538461538464,
                0.08333333333333333, 0.16666666666666666, 0.25, 0.5, 0.75, 1.0])
M, TT = np.meshgrid(m, tau)  # [9,11] (tau, m)
verf = np.vectorize(math.erf)
ncdf = lambda x: 0.5 * (1 + verf(x / math.sqrt(2)))
dt = tau[1:] - tau[:-1]
dm = m[1:] - m[:-1]
eps = 1e-8


def cs_surface(iv):
    sq = iv * np.sqrt(TT)
    d1 = (-np.log(M) + (R + 0.5 * iv ** 2) * TT) / sq
    d2 = d1 - sq
    return ncdf(d1) - M * np.exp(-R * TT) * ncdf(d2)


def violations(prices):  # prices [N,9,11] (tau,m) -> l1,l2,l3 [N]
    diff_t = prices[:, :-1, :] - prices[:, 1:, :]
    l1 = np.maximum(tau[:-1][None, :, None] * diff_t / np.maximum(dt, eps)[None, :, None], 0).sum((1, 2))
    diff_m = prices[:, :, 1:] - prices[:, :, :-1]
    slope = diff_m / np.maximum(dm, eps)[None, None, :]
    l2 = np.maximum(slope, 0).sum((1, 2))
    l3 = np.maximum(slope[:, :, :-1] - slope[:, :, 1:], 0).sum((1, 2))
    return l1, l2, l3


# ---- data: training + AD-Seq (base diffusion) + AD-Seq-ft (fine-tuned) ---
fs = sorted(glob.glob(f"{GEN_DIR}/*.npy"),
            key=lambda s: int(s.split("sample_batch")[1].split(".npy")[0]))
gen = np.concatenate([np.load(f) for f in fs], 0)
data = np.load(DATA)
n = gen.shape[0]
data = data[:n]
gl1, gl2, gl3 = violations(cs_surface(np.exp(gen[:, DAY, 0])))
tl1, tl2, tl3 = violations(cs_surface(np.exp(data[:, DAY, 0])))
ft = np.load(FT_SAMPLES)
fl1, fl2, fl3 = violations(cs_surface(np.exp(ft[:, DAY, 0])))


# ---- VolGAN next-day surfaces for the same windows ----------------------
def volgan_surfaces():
    ck = torch.load(VOLGAN_CKPT, map_location="cpu", weights_only=False)
    state = ck.get("gen_state", ck.get("gen_state_dict"))
    cfg = ck.get("config", {})
    noise_dim = int(ck.get("noise_dim", cfg.get("noise_dim", 32)))
    hidden_dim = int(ck.get("hidden_dim", cfg.get("hidden_dim", state["linear1.bias"].shape[0])))
    cond_dim = int(state["linear1.weight"].shape[1] - noise_dim)
    out_dim = int(state["linear3.weight"].shape[0])
    g = _VolGAN.Generator(noise_dim=noise_dim, cond_dim=cond_dim,
                          hidden_dim=hidden_dim, output_dim=out_dim)
    g.load_state_dict(state)
    g.eval()

    rets = data[:, :, 1, 0, 0]  # [n,22] daily log-returns (broadcast-constant)
    # current surface at t-1 = day 20, VolGAN flat order m_major_tau_minor -> [m,tau]
    logiv_tm1 = np.transpose(data[:, DAY - 1, 0], (0, 2, 1)).reshape(n, -1)  # [n,99]
    r_tm1, r_tm2 = rets[:, DAY - 1], rets[:, DAY - 2]
    rv = np.sqrt(252.0 / 21) * np.sqrt((rets[:, DAY - 21:DAY] ** 2).sum(1))  # days 0..20
    cond = np.concatenate([(np.sqrt(252) * r_tm1)[:, None], (np.sqrt(252) * r_tm2)[:, None],
                           rv[:, None], logiv_tm1], 1)  # [n,102]
    torch.manual_seed(SEED)
    noise = torch.randn(n, noise_dim)
    with torch.no_grad():
        fake = g(noise, torch.from_numpy(cond).float()).cpu().numpy()  # [n,100]
    iv_next = np.exp((logiv_tm1 + fake[:, 1:]).reshape(n, len(m), len(tau)))  # [n,m,tau]
    return np.transpose(iv_next, (0, 2, 1))  # [n,tau,m]


vl1, vl2, vl3 = violations(cs_surface(volgan_surfaces()))

# ---- plotting -----------------------------------------------------------
TICK_FS, LABEL_FS, PANEL_FS, LEG_FS = 18, 24, 28, 20
GRAY, BLUE, RED, GREEN = "#4d4d4d", "#4C72B0", "#d62728", "#2ca02c"

panels = [(r"$\ell_1$", tl1, vl1, gl1, fl1),
          (r"$\ell_2$", tl2, vl2, gl2, fl2),
          (r"$\ell_3$", tl3, vl3, gl3, fl3)]


def ccdf(x):
    x = np.sort(np.asarray(x).ravel())
    surv = 1.0 - np.arange(1, x.size + 1) / x.size
    return x, surv


y_floor = max(0.5 / n, 1e-4)

fig, axes = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
for col, (label, t_arr, v_arr, g_arr, f_arr) in enumerate(panels):
    ax = axes[col]
    series = [("training", GRAY, t_arr), ("VolGAN", BLUE, v_arr),
              ("AD-Seq", RED, g_arr), ("AD-Seq-ft", GREEN, f_arr)]
    x_max = max(float(np.asarray(d).max()) for _, _, d in series)
    x_max = max(x_max, LINTHRESH * 10)
    for name, color, arr in series:
        xs, surv = ccdf(np.asarray(arr).ravel())
        mask = surv > 0
        xs_m, surv_m = xs[mask], surv[mask]
        if xs_m.size == 0:
            continue
        xs_plot = np.append(xs_m, xs_m[-1])
        surv_plot = np.append(surv_m, y_floor)
        ax.plot(xs_plot, surv_plot, color=color, linewidth=2.5, label=name,
                drawstyle="steps-post")
        ax.plot(xs_m[-1], surv_m[-1], "o", color=color, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_yscale("log")
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_floor, 1.05)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(alpha=0.25, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(label, fontsize=PANEL_FS, pad=8)
    ax.set_xlabel("violation threshold", fontsize=LABEL_FS)
    if col == 0:
        ax.set_ylabel("P(violation > x)", fontsize=LABEL_FS)

axes[-1].legend(loc="lower left", frameon=False, fontsize=LEG_FS)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("SAVED", OUT, "n_base=", n, "n_ft=", ft.shape[0])
print("training  means", tl1.mean(), tl2.mean(), tl3.mean())
print("VolGAN    means", vl1.mean(), vl2.mean(), vl3.mean())
print("AD-Seq    means", gl1.mean(), gl2.mean(), gl3.mean())
print("AD-Seq-ft means", fl1.mean(), fl2.mean(), fl3.mean())
