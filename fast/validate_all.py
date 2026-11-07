"""
Three-way validation: original MATLAB code, scipy Python port, and fast_dem.

Step 1  make the shared design list (once):
    python validate_all.py make --out validation_designs.csv   (copy it to ../matlab/)
Step 2  run MATLAB (or Octave) on it with run_validation.m  -> matlab_results.csv
Step 3  run both Python versions and compare all three:
    python validate_all.py compare          (defaults: ../scipy_ref, ../matlab/matlab_results.csv)

The design list holds the MATLAB example, the six designs tabulated in the paper's SI
(Table S1, with the paper's KE values), and random admissible designs from each category.

Reading the "paper" column: for designs with bistable springs (homogeneous opt, and
heterogeneous pert) the KE ratio is so sensitive that the 2-decimal coefficients
printed in Table S1 are not enough to reproduce Table S2 -- moving c2/c3 by less than
+-0.005 spans e.g. 0.0117..0.062 for the homogeneous optimum. All three codes agree
with each other; the mismatch is in the rounding of the published table. See
REPLICATION.md.
"""
import os
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import fast_dem as F
from generate_dataset import sample_design_space
from validate import _ref

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER = [  # (label, c2, c3, paper KE) from SI Table S1 / S2
    ("paper: homogeneous bistable (opt)", [-116.12] * 4, [3052.97] * 4, 0.0117),
    ("paper: heterogeneous (opt)", [-84.36, -94.08, -10.11, -33.35], [1589.77, 2426.41, 1682.13, 3300.14], 0.0056),
    ("paper: het. excl. bistab. (opt)", [-59.83, -69.15, -100.21, -53.23], [2715.10, 1472.62, 3374.10, 1371.83], 0.0086),
    ("paper: homogeneous bistable (pert)", [-116.06] * 4, [2993.32] * 4, 0.062),
    ("paper: heterogeneous (pert)", [-83.57, -94.50, -10.28, -33.83], [1570.73, 2415.17, 1632.89, 3324.89], 0.036),
    ("paper: het. excl. bistab. (pert)", [-59.51, -67.46, -99.08, -53.49], [2706.04, 1465.52, 3378.14, 1384.72], 0.015),
]


def make(out, n_random, seed):
    """Write the shared design list (+ a *_labels.csv with a human-readable label per id)."""
    rows = [("matlab example (main_disorder.m)", [-70, -100, -5, -35], [1200, 2000, 1500, 3000], np.nan)]
    rows += PAPER
    rng = np.random.default_rng(seed)
    for cat in ["homo", "hetero", "hetero_excl"]:
        c2, c3 = sample_design_space(rng, n_random, cat)
        rows += [(f"random {cat}", list(c2[i]), list(c3[i]), np.nan) for i in range(n_random)]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + [f"c2_{L}" for L in "ABCD"] + [f"c3_{L}" for L in "ABCD"] + ["paper_ke"])
        for i, (_, c2, c3, p) in enumerate(rows):
            w.writerow([i] + [f"{v:.10g}" for v in c2] + [f"{v:.10g}" for v in c3] + ["nan" if np.isnan(p) else p])
    with open(out.replace(".csv", "_labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label"])
        for i, (lab, *_) in enumerate(rows):
            w.writerow([i, lab])
    print(f"wrote {len(rows)} designs -> {out}")


def _read(path):
    with open(path) as fh:
        r = csv.DictReader(fh)
        return list(r)


def compare(designs, ref_dir, matlab, workers):
    """Run fast_dem and the scipy reference on every design and tabulate against MATLAB/paper."""
    D = _read(designs)
    try:
        labels = {int(r["id"]): r["label"] for r in _read(designs.replace(".csv", "_labels.csv"))}
    except FileNotFoundError:
        labels = {}
    ids = np.array([int(r["id"]) for r in D])
    c2 = np.array([[float(r[f"c2_{L}"]) for L in "ABCD"] for r in D])
    c3 = np.array([[float(r[f"c3_{L}"]) for L in "ABCD"] for r in D])
    paper = np.array([float(r["paper_ke"]) for r in D])

    fast = F.main_disorder_v2(c2, c3)
    with ProcessPoolExecutor(workers) as ex:
        scipy = np.array(list(ex.map(_ref, [(ref_dir, c2[i], c3[i]) for i in range(len(ids))])))
    mat = np.full(len(ids), np.nan)
    engine = ""
    if matlab and not os.path.exists(matlab):
        print(f"(no MATLAB results at {matlab}; comparing the Python codes only)")
        matlab = None
    if matlab:
        M = {int(r["id"]): r for r in _read(matlab)}
        mat = np.array([float(M[i]["ratio"]) if i in M else np.nan for i in ids])
        engine = next(iter(M.values()))["engine"] if M else ""

    def rel(a, b):
        return np.abs(a / b - 1)

    mname = f"MATLAB[{engine}]" if matlab else "MATLAB"
    print(f"{'id':>3} {'design':36s} {mname:>16s} {'scipy':>12s} {'fast':>12s} {'paper':>8s}"
          f"  {'scipy/MATLAB':>12s} {'fast/scipy':>10s} {'fast/MATLAB':>11s}")
    for k, i in enumerate(ids):
        p = "" if np.isnan(paper[k]) else f"{paper[k]:.4f}"
        print(f"{i:3d} {labels.get(i, '')[:36]:36s} {mat[k]:16.8f} {scipy[k]:12.8f} {fast[k]:12.8f} {p:>8s}"
              f"  {rel(scipy[k], mat[k]):12.1e} {rel(fast[k], scipy[k]):10.1e} {rel(fast[k], mat[k]):11.1e}")
    print()
    for name, a, b in [("scipy vs MATLAB", scipy, mat), ("fast vs scipy", fast, scipy), ("fast vs MATLAB", fast, mat)]:
        e = rel(a, b)
        if np.isfinite(e).any():
            print(f"{name:16s} median rel. diff {np.nanmedian(e):.1e}   max {np.nanmax(e):.1e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default="validation_designs.csv")
    m.add_argument("--n-random", type=int, default=3, help="random designs per category")
    m.add_argument("--seed", type=int, default=2026)
    c = sub.add_parser("compare")
    c.add_argument("--designs", default=os.path.join(HERE, "validation_designs.csv"))
    c.add_argument("--ref-dir", default=os.path.join(HERE, "..", "scipy_ref"))
    c.add_argument("--matlab", default=os.path.join(HERE, "..", "matlab", "matlab_results.csv"))
    c.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out, a.n_random, a.seed)
    else:
        compare(a.designs, a.ref_dir, a.matlab, a.workers)


if __name__ == "__main__":
    main()
