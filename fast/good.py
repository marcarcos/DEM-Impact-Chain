"""
Robustness of the good designs in a dataset (paper Fig. 1d style).

    python robustness_of_good.py --data dataset.npz --threshold 0.03 --k 100 --eps 0.05

For every nominal design with KE ratio <= threshold, run k perturbed copies
(c = c*(1+delta), delta ~ N(0,(eps/2)^2), inadmissible copies resampled) and report
mu_KE per category, plus P(mu_KE < threshold). Saves robust_good.npz.
"""
import argparse

import numpy as np

import fast_dem as F
from generate_dataset import perturb

NAMES = ["homogeneous", "heterogeneous", "het. excl. bistability"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="robust_good.npz")
    a = ap.parse_args()

    d = np.load(a.data)
    good = d["ratio"] <= a.threshold
    c2, c3, cat, nominal = d["c2"][good], d["c3"][good], d["category"][good], d["ratio"][good]
    n = len(c2)
    print(f"{n} designs with KE <= {a.threshold}; running {n * a.k} perturbed simulations")

    rng = np.random.default_rng(a.seed)
    P2 = np.empty((n * a.k, 4))
    P3 = np.empty((n * a.k, 4))
    for c in range(3):
        idx = np.flatnonzero(cat == c)
        if len(idx) == 0:
            continue
        p2, p3 = perturb(rng, c2[idx], c3[idx], a.k, a.eps, homogeneous=(c == 0))
        rows = (idx[:, None] * a.k + np.arange(a.k)).ravel()
        P2[rows], P3[rows] = p2, p3

    ratio = F.main_disorder_v2(P2, P3).reshape(n, a.k)
    mu = np.nanmean(ratio, axis=1)
    sd = np.nanstd(ratio, axis=1)

    print(f"\n{'category':24s} {'designs':>7s} {'median nominal':>15s} {'median mu_KE':>13s} "
          f"{'P(mu_KE<thr)':>13s}")
    for c in range(3):
        m = cat == c
        if m.any():
            print(f"{NAMES[c]:24s} {m.sum():7d} {np.median(nominal[m]):15.4f} {np.median(mu[m]):13.4f} "
                  f"{100 * (mu[m] < a.threshold).mean():12.1f}%")
    np.savez(a.out, c2=c2, c3=c3, category=cat, nominal=nominal, perturbed=ratio,
             mu_ke=mu, sd_ke=sd, eps=a.eps, k=a.k, threshold=a.threshold)
    print(f"\nsaved -> {a.out}")


    r = np.load("robust_good.npz")
    nom, mu, cat = r["nominal"], r["mu_ke"], r["category"]
    bins = np.linspace(0.01, 0.03, 5)            # 4 equal-width bins
    rng = np.random.default_rng(0)
    for c, name in enumerate(["homo", "hetero", "hetero_excl"]):
        m = (cat == c) & (nom >= 0.01)
        b = np.digitize(nom[m], bins) - 1
        k = np.bincount(b, minlength=4)[:4].min()  # same count from every bin
        idx = np.concatenate([rng.choice(np.flatnonzero(b == j), k, replace=False) for j in range(4)])
        sel = mu[m][idx]
        print(f"{name:12s} n={len(sel):4d}  median mu_KE {np.median(sel):.4f}  P(<0.03) {100*(sel<0.03).mean():.1f}%")


if __name__ == "__main__":
    main()