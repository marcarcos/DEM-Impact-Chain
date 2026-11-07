"""
One-command pass/fail check that the fast simulator reproduces MATLAB and the paper.

    cd fast && python check_simulation.py          # ~10-20 s, needs only numpy + numba

Checks (all against files already in the repo, MATLAB/scipy not needed):
  1. Equations of motion: fast_dem._rhs vs MATLAB on 5000 random states/designs
     (../matlab/rhs_inputs.csv -> rhs_matlab.csv).
  2. KE ratio: fast_dem vs MATLAB's main_disorder_fn on the 16 validation designs
     (../matlab/matlab_results.csv).
  3. Paper SI Table S2: KE ratios of the published designs. Two of the six are
     skipped because their bistable springs make the KE ratio change by up to 5x
     within the +-0.005 rounding of the published coefficients (see REPLICATION.md).
  4. Paper Fig. 4 / Sec. 3: impactor mass perturbed to 1.01 M -> linear peak KE
     rises 1.02x and KE ratios are 11 %, 2.6 %, 0.8 % for the three optimised chains.

Exit code 0 if everything passes, 1 otherwise.
"""
import os
import sys

import numpy as np

import fast_dem as F

HERE = os.path.dirname(os.path.abspath(__file__))
MATLAB_DIR = os.path.join(HERE, "..", "matlab")

# Paper SI Table S1 coefficients (c2 A-D, c3 A-D) and Table S2 / main-text KE values
OPT = {
    "homogeneous bistable": ([-116.12] * 4, [3052.97] * 4),
    "heterogeneous": ([-84.36, -94.08, -10.11, -33.35], [1589.77, 2426.41, 1682.13, 3300.14]),
    "het. excl. bistability": ([-59.83, -69.15, -100.21, -53.23], [2715.10, 1472.62, 3374.10, 1371.83]),
}
PERT = {
    "homogeneous bistable": ([-116.06] * 4, [2993.32] * 4),
    "heterogeneous": ([-83.57, -94.50, -10.28, -33.83], [1570.73, 2415.17, 1632.89, 3324.89]),
    "het. excl. bistability": ([-59.51, -67.46, -99.08, -53.49], [2706.04, 1465.52, 3378.14, 1384.72]),
}
# (name, design, paper KE, check?) -- False = not reproducible from 2-decimal coefficients
TABLE_S2 = [
    ("opt  homogeneous bistable", OPT["homogeneous bistable"], 0.0117, False),
    ("opt  heterogeneous", OPT["heterogeneous"], 0.0056, True),
    ("opt  het. excl. bistability", OPT["het. excl. bistability"], 0.0086, True),
    ("pert homogeneous bistable", PERT["homogeneous bistable"], 0.062, True),
    ("pert heterogeneous", PERT["heterogeneous"], 0.036, False),
    ("pert het. excl. bistability", PERT["het. excl. bistability"], 0.015, True),
]
FIG4 = {"homogeneous bistable": 0.11, "heterogeneous": 0.026, "het. excl. bistability": 0.008}

failures = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")
    if not ok:
        failures.append(name)


def rhs_vs_matlab():
    print("1. equations of motion vs MATLAB")
    A = np.loadtxt(os.path.join(MATLAB_DIR, "rhs_inputs.csv"), delimiter=",")
    ref = np.loadtxt(os.path.join(MATLAB_DIR, "rhs_matlab.csv"), delimiter=",")
    nm = F.N + 1
    out = np.empty(2 * nm)
    worst = 0.0
    for row, r in zip(A, ref):
        x = row[8:]
        y = np.concatenate([x[0::2], x[1::2]])          # interleaved -> [u..., v...]
        F._rhs(y, np.ones(F.N), np.repeat(row[:4], 5), np.repeat(row[4:8], 5),
               F.M_LAYER, F.M_IMPACTOR, F.C_IMPACT, F.ZETA, out)
        mine = np.empty(2 * nm)
        mine[0::2], mine[1::2] = out[:nm], out[nm:]
        scale = max(np.abs(r).max(), 1.0)
        worst = max(worst, np.abs(mine - r).max() / scale)
    check(f"RHS on {len(A)} random states", worst < 1e-10, f"max scaled diff {worst:.1e}")


def ke_vs_matlab():
    print("2. KE ratio vs MATLAB (ode45, rtol = atol = 1e-12)")
    D = np.loadtxt(os.path.join(MATLAB_DIR, "validation_designs.csv"), delimiter=",", skiprows=1)
    M = np.genfromtxt(os.path.join(MATLAB_DIR, "matlab_results.csv"), delimiter=",",
                      names=True, dtype=None, encoding=None)
    mat = dict(zip(M["id"].astype(int), M["ratio"]))
    r = F.main_disorder_v2(D[:, 1:5], D[:, 5:9])
    rel = np.array([abs(r[k] / mat[int(i)] - 1) for k, i in enumerate(D[:, 0])])
    check(f"{len(D)} designs", rel.max() < 1e-4,
          f"median rel. diff {np.median(rel):.1e}, max {rel.max():.1e}")


def table_s2():
    print("3. paper SI Table S2 (KE ratio at the design point)")
    c2 = np.array([d[0] for _, d, _, _ in TABLE_S2])
    c3 = np.array([d[1] for _, d, _, _ in TABLE_S2])
    r = F.main_disorder_v2(c2, c3)
    for (name, _, paper, use), v in zip(TABLE_S2, r):
        if use:
            check(name, abs(v / paper - 1) < 0.05, f"sim {v:.4f}  paper {paper:.4f}")
        else:
            print(f"  [SKIP] {name:34s} sim {v:.4f}  paper {paper:.4f}  (coefficient rounding)")


def fig4():
    print("4. paper Fig. 4: impactor mass 1.01 M")
    m = 1.01 * F.M_IMPACTOR
    inc = F.linear_max_ke(m_imp=m) / F.linear_max_ke()
    check("linear peak KE increase", abs(inc - 1.02) < 0.005, f"sim x{inc:.3f}  paper x1.02")
    c2 = np.array([OPT[k][0] for k in FIG4])
    c3 = np.array([OPT[k][1] for k in FIG4])
    r = F.main_disorder_v2(c2, c3, m_imp=m)
    for (name, paper), v in zip(FIG4.items(), r):
        # paper quotes 2 significant figures (11 %, 2.6 %, 0.8 %)
        check(name, abs(v / paper - 1) < 0.06, f"sim {v:.4f}  paper {paper:.3f}")


if __name__ == "__main__":
    rhs_vs_matlab()
    ke_vs_matlab()
    table_s2()
    fig4()
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
