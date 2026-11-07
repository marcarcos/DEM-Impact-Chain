"""
Robustness of the well-performing designs in a dataset (paper Fig. 1d / SI-4 protocol).

    python good.py --data dataset.npz                         # Fig. 1d: KE in [0.01, 0.03]
    python good.py --data dataset.npz --lo 0.01 --threshold 0.10 --bins 9 \
                   --per-category 900 --out robust_si4.npz    # SI-4 / Fig. S1 variant

Protocol (main text Sec. 2.2, "ensemble-level comparison"):
  1. Baseline set. From the NOMINAL designs of each category (homogeneous,
     heterogeneous, heterogeneous-excluding-bistability) keep those with
     KE ratio in [lo, threshold], split that interval into ``--bins`` equal-width bins
     and draw the same number of designs from every bin, so the nominal KE values are
     approximately uniform (paper: 200 per category over [0.01, 0.03]). If a bin is
     short, every bin is cut to the smallest bin count (a warning is printed).
  2. Design uncertainty. Each baseline design gets ``--k`` Monte Carlo copies
     c = c* (1 + delta), delta ~ N(0, (eps/2)^2) (paper Eq. 4; ~95 % have |delta| <= eps),
     one delta per spring family (one for the whole chain if homogeneous), inadmissible
     copies (W < 0, SI-2) are resampled. Paper: k = 100, eps = 5 %.
  3. Report, per category, the median of mu_KE (mean perturbed KE ratio of a design)
     and P(mu_KE < threshold). Paper Fig. 1d: medians 0.105 / 0.067 / 0.050 and
     P = 0 % / 0 % / 9 % for homogeneous / heterogeneous / het.-excl.-bistability.

REPLICATION CAVEAT: the paper's baseline designs come from gradient-based optimisation
(fmincon from random starts), while dataset.npz holds uniform random samples of the
design space. Random samples that land in [0.01, 0.03] need not sit in the same
(sharp) minima as optimised ones, so their robustness can differ. See REPLICATION.md.

Output (npz): baseline c2/c3/category/nominal KE, the (n, k) perturbed KE ratios,
mu_KE, sd_KE and the run settings.
"""
import argparse

import numpy as np

import fast_dem as F
from generate_dataset import CATEGORIES, perturb

NAMES = {0: "homogeneous", 1: "heterogeneous", 2: "het. excl. bistability"}


def select_baseline(rng, ratio, category, is_nominal, lo, hi, n_bins, per_category):
    """Indices of a baseline set with ~uniform nominal KE in [lo, hi] for every category.

    Returns a dict {category code: index array into the dataset}.
    """
    edges = np.linspace(lo, hi, n_bins + 1)
    chosen = {}
    for code in CATEGORIES.values():
        ok = (category == code) & is_nominal & (ratio >= lo) & (ratio <= hi)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            print(f"warning: no {NAMES[code]} designs with KE in [{lo}, {hi}]")
            chosen[code] = idx
            continue
        b = np.clip(np.digitize(ratio[idx], edges) - 1, 0, n_bins - 1)  # hi falls in last bin
        counts = np.bincount(b, minlength=n_bins)
        want = per_category // n_bins
        take = min(want, counts.min())
        if take < want:
            print(f"warning: {NAMES[code]}: bin counts {counts.tolist()} -> only {take} per bin "
                  f"({take * n_bins} designs instead of {want * n_bins})")
        chosen[code] = np.concatenate(
            [rng.choice(idx[b == j], take, replace=False) for j in range(n_bins)]).astype(int)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="dataset.npz", help="output of generate_dataset.py")
    ap.add_argument("--lo", type=float, default=0.01, help="lower edge of the baseline KE window")
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="upper edge of the baseline KE window and the target for P(mu_KE < thr)")
    ap.add_argument("--bins", type=int, default=4, help="equal-width bins for the uniform baseline")
    ap.add_argument("--per-category", type=int, default=200, help="baseline designs per category")
    ap.add_argument("--k", type=int, default=100, help="Monte Carlo copies per baseline design")
    ap.add_argument("--eps", type=float, default=0.05, help="noise level epsilon (sigma = eps/2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="robust_good.npz")
    a = ap.parse_args()

    d = np.load(a.data)
    is_nominal = d["is_nominal"] if "is_nominal" in d.files else np.ones(len(d["ratio"]), bool)
    rng = np.random.default_rng(a.seed)
    chosen = select_baseline(rng, d["ratio"], d["category"], is_nominal,
                             a.lo, a.threshold, a.bins, a.per_category)
    sel = np.concatenate([chosen[c] for c in sorted(chosen)])
    c2, c3, cat, nominal = d["c2"][sel], d["c3"][sel], d["category"][sel], d["ratio"][sel]
    n = len(sel)
    print(f"{n} baseline designs with KE in [{a.lo}, {a.threshold}]; "
          f"running {n * a.k} perturbed simulations (eps = {a.eps:g})")

    # k perturbed copies per baseline design; rows of design i are i*k .. i*k + k-1
    P2 = np.empty((n * a.k, 4))
    P3 = np.empty((n * a.k, 4))
    for code in CATEGORIES.values():
        idx = np.flatnonzero(cat == code)
        if len(idx) == 0:
            continue
        p2, p3 = perturb(rng, c2[idx], c3[idx], a.k, a.eps, homogeneous=(code == CATEGORIES["homo"]))
        rows = (idx[:, None] * a.k + np.arange(a.k)).ravel()
        P2[rows], P3[rows] = p2, p3

    ratio = F.main_disorder_v2(P2, P3).reshape(n, a.k)
    n_fail = int(np.isnan(ratio).sum())
    if n_fail:
        print(f"warning: {n_fail} perturbed simulations failed (NaN, ignored in the means)")
    mu = np.nanmean(ratio, axis=1)
    sd = np.nanstd(ratio, axis=1)

    print(f"\n{'category':24s} {'designs':>7s} {'median nominal':>15s} {'median mu_KE':>13s} "
          f"{'P(mu_KE<thr)':>13s}")
    for code in CATEGORIES.values():
        m = cat == code
        if m.any():
            print(f"{NAMES[code]:24s} {m.sum():7d} {np.median(nominal[m]):15.4f} "
                  f"{np.median(mu[m]):13.4f} {100 * (mu[m] < a.threshold).mean():12.1f}%")
    print("paper Fig. 1d (eps 5%, [0.01, 0.03]): median mu_KE 0.105 / 0.067 / 0.050, "
          "P(mu_KE < 0.03) 0% / 0% / 9%")
    np.savez(a.out, c2=c2, c3=c3, category=cat, nominal=nominal, perturbed=ratio,
             mu_ke=mu, sd_ke=sd, eps=a.eps, k=a.k, lo=a.lo, threshold=a.threshold,
             bins=a.bins, seed=a.seed, source_index=sel)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
