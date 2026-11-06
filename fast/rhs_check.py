"""
Quick validation of the model itself: evaluate the equations of motion (the ODE right-hand
side) at many random states and designs in all three codes and compare.

    python rhs_check.py make --n 5000                  # writes rhs_inputs.csv
    octave rhs_check.m   (or run rhs_check in MATLAB)  # writes rhs_matlab.csv  (seconds)
    python rhs_check.py compare --ref-dir ../scipy_ref # compares MATLAB, scipy port, fast_dem

States are random but chosen to hit every branch: springs in tension and compression,
deep compression past the bistable well, impactor overlapping or separated from mass 1.
"""
import argparse
import os
import sys

import numpy as np

import fast_dem as F
from generate_dataset import sample_design_space

NM = F.N + 1  # 21 masses


def make(n, seed, out):
    rng = np.random.default_rng(seed)
    third = n // 3
    parts = [sample_design_space(rng, k, c) for k, c in
             [(third, "homo"), (third, "hetero"), (n - 2 * third, "hetero_excl")]]
    c2 = np.vstack([p[0] for p in parts])
    c3 = np.vstack([p[1] for p in parts])
    # positions: random walk so neighbour differences span tension and compression,
    # with some large compressions (past the bistable roots ~0.01-0.1)
    steps = rng.normal(0, 0.02, (n, NM)) * rng.choice([0.2, 1.0, 3.0], (n, 1))
    u = np.cumsum(steps, axis=1)
    u -= u[:, -1:] * rng.uniform(0, 1, (n, 1))     # last mass sometimes left/right of the wall
    flip = rng.random(n) < 0.5                       # half the rows: impactor overlapping mass 1
    u[flip, 0] = u[flip, 1] + rng.uniform(0, 0.005, flip.sum())
    u[~flip, 0] = u[~flip, 1] - rng.uniform(0, 0.005, (~flip).sum())
    v = rng.normal(0, 0.2, (n, NM))
    x = np.empty((n, 2 * NM))
    x[:, 0::2] = u   # interleaved [u0, v0, u1, v1, ...] as in MATLAB / scipy code
    x[:, 1::2] = v
    np.savetxt(out, np.hstack([c2, c3, x]), delimiter=",", fmt="%.17g")
    print(f"wrote {n} rows -> {out}")


def compare(inp, matlab_out, ref_dir):
    A = np.loadtxt(inp, delimiter=",")
    c2, c3, x = A[:, :4], A[:, 4:8], A[:, 8:]
    n = len(A)

    # scipy port (original Python)
    sys.path.insert(0, os.path.abspath(ref_dir))
    for m in ("impact_equation_of_motion_hetero", "impact_force"):
        sys.modules.pop(m, None)
    from impact_equation_of_motion_hetero import impact_equation_of_motion_hetero as ref_rhs
    mat_info = np.array([F.M_LAYER, F.M_IMPACTOR, F.C_IMPACT, F.ZETA])
    dx_py = np.empty_like(x)
    for i in range(n):
        info = np.column_stack((np.ones(20), np.repeat(c2[i], 5), np.repeat(c3[i], 5)))
        dx_py[i] = ref_rhs(0.0, x[i], info, mat_info, 0.0)

    # fast_dem (state layout [u0..uN, v0..vN])
    dx_fast = np.empty_like(x)
    out = np.empty(2 * NM)
    for i in range(n):
        y = np.concatenate([x[i, 0::2], x[i, 1::2]])
        C2 = np.repeat(c2[i], 5)
        C3 = np.repeat(c3[i], 5)
        F._rhs(y, np.ones(20), C2, C3, F.M_LAYER, F.M_IMPACTOR, F.C_IMPACT, F.ZETA, out)
        dx_fast[i, 0::2] = out[:NM]
        dx_fast[i, 1::2] = out[NM:]

    def report(name, a, b):
        scale = np.maximum(np.abs(b), 1e-12)
        err = np.abs(a - b) / scale
        acc = np.abs(a - b)[:, 1::2]
        print(f"{name:22s} max rel. diff {err.max():.1e}   max abs. diff in accelerations {acc.max():.1e}")

    print(f"{n} random states; accelerations range up to {np.abs(dx_py[:, 1::2]).max():.1e}")
    report("fast_dem vs scipy", dx_fast, dx_py)
    if matlab_out and os.path.exists(matlab_out):
        dx_m = np.loadtxt(matlab_out, delimiter=",")
        report("scipy vs MATLAB", dx_py, dx_m)
        report("fast_dem vs MATLAB", dx_fast, dx_m)
    else:
        print("(no MATLAB output found; run rhs_check.m to include it)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n", type=int, default=5000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default="rhs_inputs.csv")
    c = sub.add_parser("compare")
    c.add_argument("--inputs", default="rhs_inputs.csv")
    c.add_argument("--matlab", default="rhs_matlab.csv")
    c.add_argument("--ref-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.n, a.seed, a.out)
    else:
        compare(a.inputs, a.matlab, a.ref_dir)


if __name__ == "__main__":
    main()
