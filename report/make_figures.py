"""
Regenerate every figure and number used in report/checkpoint.tex.

    cd report && python make_figures.py        # ~2-3 min on 4 cores

Writes figures/*.pdf and figures/numbers.json. Everything comes from the verified
simulator (fast/fast_dem.py), the existing random-design dataset (fast/dataset.npz)
and the stored one-at-a-time sweep (scipy_ref/data/sensitivity_*.csv).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fast"))
import fast_dem as F                                   # noqa: E402
from check_simulation import OPT                       # noqa: E402
from generate_dataset import perturb, CATEGORIES       # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Reference categorical palette (fixed order) + text/grid tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT_NAMES = ["Homogeneous bistable", "Heterogeneous", "Het. excl. bistability"]
CAT_KEYS = list(OPT)                                   # same order as CAT_NAMES
MARK = ["o", "s", "^", "D"]
STYLE = ["-", "--", "-.", ":"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
numbers = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")), dpi=110)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. OAT sweep
def oat():
    path = sorted(glob.glob(os.path.join(HERE, "..", "scipy_ref", "data", "sensitivity_*.csv")))[-1]
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    base = F.main_disorder_v2([-70, -100, -5, -35], [1200, 2000, 1500, 3000])
    numbers["oat_baseline"] = float(base)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, coef in zip(axes, ["c2", "c3"]):
        for f in range(4):
            m = d["varied_parameter"] == f"{coef}_{f + 1}"
            x, y = d["percent_change"][m], d["KE_ratio"][m]
            c2 = d[f"c2_{f + 1}"][m]
            c3 = d[f"c3_{f + 1}"][m]
            adm = c3 > (2 / 9) * c2 ** 2
            o = np.argsort(x)
            x, y, adm = x[o], y[o], adm[o]
            ax.plot(x, y, STYLE[f], color=C[f], label=f"family {'ABCD'[f]}")
            ax.plot(x[adm], y[adm], MARK[f], color=C[f], ms=3.5, ls="none")
            ax.plot(x[~adm], y[~adm], MARK[f], mfc="white", mec=C[f], ms=4, ls="none")
        ax.axhline(base, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel(f"change in ${coef[0]}_{coef[1]}$ of one family (%)")
        ax.set_title(f"varying ${coef[0]}_{coef[1]}$", fontsize=9, color=INK)
    axes[0].set_ylabel("KE ratio")
    axes[0].legend(loc="upper left", ncol=2)
    numbers["oat_c3A_minus50"] = float(d["KE_ratio"][(d["varied_parameter"] == "c3_1")
                                                     & np.isclose(d["percent_change"], -50)][0])
    rng_c2 = d["KE_ratio"][np.char.startswith(d["varied_parameter"].astype(str), "c2")]
    numbers["oat_c2_range"] = [float(rng_c2.min()), float(rng_c2.max())]
    save(fig, "oat_sensitivity.pdf")


# ------------------------------------------------- 2. impact-condition slices
def impact_slices():
    n = 401
    mm = np.linspace(0.009, 0.011, n) * F.M0_IMPACTOR
    vv = np.linspace(0.2, 0.3, n)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for k, key in enumerate(CAT_KEYS):
        c2, c3 = np.array(OPT[key][0], float), np.array(OPT[key][1], float)
        rm = [F.main_disorder_v2(c2, c3, m_imp=m) for m in mm]
        rv = [F.main_disorder_v2(c2, c3, v_imp=v) for v in vv]
        axes[0].plot(mm / F.M0_IMPACTOR, rm, color=C[k], lw=1.1, label=CAT_NAMES[k])
        axes[1].plot(vv, rv, color=C[k], lw=1.1)
        # mean |d KE / d(M/M0)| and |d KE / d(V/V0)| (paper Fig. 2c measures sensitivity
        # by these derivatives); finite differences on the 401-point slices
        numbers[f"slice_dM_{k}"] = float(np.mean(np.abs(np.diff(rm) / np.diff(mm / F.M0_IMPACTOR))))
        numbers[f"slice_dV_{k}"] = float(np.mean(np.abs(np.diff(rv) / np.diff(vv))))
        numbers[f"slice_range_M_{k}"] = [float(np.min(rm)), float(np.max(rm))]
        numbers[f"slice_range_V_{k}"] = [float(np.min(rv)), float(np.max(rv))]
    for ax in axes:
        ax.axvline(0.01 if ax is axes[0] else 0.25, color=INK2, lw=0.8, ls=":")
    axes[0].set_xlabel("$M/M_0$  ($V/V_0=0.25$)")
    axes[1].set_xlabel("$V/V_0$  ($M/M_0=0.01$)")
    axes[0].set_ylabel("KE ratio")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, "impact_slices.pdf")


# ---------------------------------------------- 3. noise-level sweep (Fig. 1c)
def noise_sweep():
    eps = np.linspace(0, 0.05, 21)       # 0, 0.25, ..., 5 %, as in the paper
    ns = 500
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    res = {}
    for k, key in enumerate(CAT_KEYS):
        c2 = np.array([OPT[key][0]], float)
        c3 = np.array([OPT[key][1]], float)
        mu, sd = [], []
        for e in eps:
            if e == 0:
                r = np.atleast_1d(F.main_disorder_v2(c2, c3))
            else:
                p2, p3 = perturb(rng, c2, c3, ns, e, homogeneous=(k == 0))
                r = F.main_disorder_v2(p2, p3)
            mu.append(np.nanmean(r))
            sd.append(np.nanstd(r))
        mu, sd = np.array(mu), np.array(sd)
        res[key] = (mu, sd)
        ax.fill_between(eps * 100, np.maximum(mu - sd, 0), mu + sd, color=C[k], alpha=0.15, lw=0)
        ax.plot(eps * 100, mu, STYLE[k], marker=MARK[k], ms=3.5, color=C[k], label=CAT_NAMES[k])
        numbers[f"noise_mu0_{k}"] = float(mu[0])
        numbers[f"noise_mu5_{k}"] = float(mu[-1])
        numbers[f"noise_mu025_{k}"] = float(mu[1])
        numbers[f"noise_sd5_{k}"] = float(sd[-1])
    ax.set_xlabel(r"noise level $\epsilon$ (%)")
    ax.set_ylabel(r"mean KE ratio ($\pm\sigma$)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, "noise_sweep.pdf")
    numbers["noise_samples"] = ns


# --------------------------------------------- 4. dataset KE distribution
def dataset_hist():
    d = np.load(os.path.join(HERE, "..", "fast", "dataset.npz"))
    r, cat = d["ratio"], d["category"]
    numbers["data_rows"] = int(len(r))
    numbers["data_failed"] = int(np.isnan(r).sum())
    numbers["data_linear_max_ke"] = float(d["linear_max_ke"])
    bins = np.logspace(-2.5, 1, 71)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k in range(3):
        x = r[(cat == k) & np.isfinite(r)]
        ax.hist(x, bins=bins, histtype="step", lw=1.4, ls=STYLE[k], color=C[k],
                weights=np.full(len(x), 100 / len(x)), label=CAT_NAMES[k].replace(" bistable", ""))
        numbers[f"data_n_{k}"] = int(len(x))
        numbers[f"data_median_{k}"] = float(np.median(x))
        numbers[f"data_min_{k}"] = float(x.min())
        numbers[f"data_frac_below1_{k}"] = float(np.mean(x < 1))
        numbers[f"data_n_window_{k}"] = int(np.sum((x >= 0.01) & (x <= 0.03)))
        numbers[f"data_n_le003_{k}"] = int(np.sum(x <= 0.03))
    ax.axvline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("KE ratio (nominal designs)")
    ax.set_ylabel("designs per bin (%)")
    ax.legend(loc="upper left")
    save(fig, "dataset_hist.pdf")
    t = d["spring_type"]
    for code, nm in enumerate(["monotonic", "snap", "bistable"]):
        numbers[f"data_type_{nm}"] = float((t == code).mean())


# ----------------------------------------- 5. preliminary robustness (Fig. 1d)
def robustness():
    d = np.load(os.path.join(HERE, "..", "fast", "dataset.npz"))
    r, cat = d["ratio"], d["category"]
    rng = np.random.default_rng(1)
    k, eps, lo, hi, nb = 100, 0.05, 0.01, 0.03, 4
    edges = np.linspace(lo, hi, nb + 1)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    hb = np.linspace(0, 0.2, 41)
    for c in range(3):
        idx = np.flatnonzero((cat == c) & (r >= lo) & (r <= hi))
        b = np.clip(np.digitize(r[idx], edges) - 1, 0, nb - 1)
        take = np.bincount(b, minlength=nb).min()
        sel = np.concatenate([rng.choice(idx[b == j], take, replace=False) for j in range(nb)])
        p2, p3 = perturb(rng, d["c2"][sel], d["c3"][sel], k, eps, homogeneous=(c == 0))
        mu = np.nanmean(F.main_disorder_v2(p2, p3).reshape(len(sel), k), axis=1)
        ax.hist(mu, bins=hb, histtype="step", lw=1.4, ls=STYLE[c], color=C[c],
                weights=np.full(len(mu), 100 / len(mu)), label=CAT_NAMES[c].replace(" bistable", ""))
        numbers[f"rob_n_{c}"] = int(len(sel))
        numbers[f"rob_median_{c}"] = float(np.median(mu))
        numbers[f"rob_p_{c}"] = float(np.mean(mu < hi))
    ax.axvline(hi, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel(r"$\mu_{KE}$ (mean KE ratio, 5% noise, 100 trials)")
    ax.set_ylabel("designs per bin (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    save(fig, "robustness_prelim.pdf")


# ----------------------------------------------- 6. validation numbers
def validation():
    vals = {}
    for key in CAT_KEYS:
        vals[key] = float(F.main_disorder_v2(np.array(OPT[key][0], float), np.array(OPT[key][1], float)))
    numbers["opt_sim"] = vals


if __name__ == "__main__":
    F.linear_max_ke()
    validation()
    oat()
    dataset_hist()
    impact_slices()
    noise_sweep()
    robustness()
    with open(os.path.join(OUT, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1)
    print(json.dumps(numbers, indent=1))
