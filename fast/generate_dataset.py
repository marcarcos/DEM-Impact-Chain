"""
Generate a dataset of (c2, c3) -> max KE ratio.

Default (--sampler paper) follows the design space of Tamur et al.:
  * each layer: c2 in [-120, 0], c3 in [0, 4000], with positive strain energy c3 > (2/9) c2^2
  * --category hetero       : 4 independent layers                   ("heterogeneous")
               hetero_excl  : 4 layers, also c2^2 <= 4 c3 (no bistable)  ("heterogeneous excluding bistability")
               homo         : one (c2, c3) shared by all 4 layers      ("homogeneous")
               mixed        : equal thirds of the three (default)
  * --perturb K adds K perturbed copies of every nominal design, c = c*(1 + delta),
    delta ~ N(0, (eps/2)^2) (so ~95% have |delta| <= eps), one delta per layer
    (one for the whole chain when homogeneous), resampling any copy with c3 <= (2/9) c2^2.
    Rows sharing design_id form one Monte Carlo ensemble -> robustness labels
    (mean / std of the KE ratio), as in the paper's Fig. 1c,d.

    python generate_dataset.py --rows 360000 --out dataset.npz                       # 360k nominal designs
    python generate_dataset.py --rows 360000 --perturb 99 --eps 0.05 --out robust.npz # 3,600 designs x 100

--sampler local keeps the old behaviour (baseline x uniform factor in [1-spread, 1+spread]).

Output npz keys
  c2, c3        (rows, 4)  family coefficients (A, B, C, D from the impacted end)
  ratio         (rows,)    KE ratio max KE_non / max KE_lin at the last mass (NaN = failed)
  steps, status (rows,)    accepted RK45 steps / 0 ok, 1 max_steps, 2 step underflow
  design_id     (rows,)    rows with the same id are one design + its perturbed copies
  is_nominal    (rows,)    True for the unperturbed design of each id
  category      (rows,)    0 homo, 1 hetero, 2 hetero_excl (-1 for --sampler local)
  spring_type   (rows, 4)  per-family SI-3 type, see SPRING_TYPES
  plus scalars: linear_max_ke, rtol, atol, sampler, perturb, eps, spread, seed

REPLICATION NOTE: the paper builds its baseline sets by OPTIMISING (fmincon from random
starts) and keeping solutions with KE in a window. This script draws designs uniformly
at random from the admissible design space; good designs found this way are a
different population from optimised minima. The inverse-design step is not
implemented yet (see REPLICATION.md).
"""
import argparse
import os
import time

import numpy as np
import numba

import fast_dem as F

# Example design from matlab/main_disorder.m (centre of the old "local" sampler)
BASE_C2 = np.array([-70.0, -100.0, -5.0, -35.0])
BASE_C3 = np.array([1200.0, 2000.0, 1500.0, 3000.0])

# Design space of the paper (Methods 5.2)
C2_RANGE = (-120.0, 0.0)
C3_RANGE = (0.0, 4000.0)
# Chain categories (paper Sec. 2.1): homogeneous / heterogeneous / heterogeneous
# excluding bistability
CATEGORIES = {"homo": 0, "hetero": 1, "hetero_excl": 2}
# Per-family spring type codes (SI-3), with s = c2^2:
#   monotonic     s <= 3 c3              f'(Dx) > 0 for all Dx > 0         (Eq. S19)
#   snap-through  3 c3 < s <= 4 c3       local max/min, no positive root   (Eq. S24-S25)
#   bistable      s > 4 c3               two positive roots of f           (Eq. S23)
#   inadmissible  c3 <= (2/9) s          stored energy W < 0 somewhere     (Eq. S14)
# (the paper writes S25 as s < 4 c3; the boundary s = 4 c3 has measure zero and we put
#  it with snap-through, consistent with the hetero_excl constraint c2^2 <= 4 c3.)
SPRING_TYPES = {0: "monotonic", 1: "snap-through", 2: "bistable", 3: "inadmissible"}


def sample_coeffs(rng, n, spread):
    """Old local sampler: baseline times uniform factors (ignores the W >= 0 constraint)."""
    c2 = BASE_C2 * rng.uniform(1 - spread, 1 + spread, size=(n, 4))
    c3 = BASE_C3 * rng.uniform(1 - spread, 1 + spread, size=(n, 4))
    return c2, c3


def positive_energy(c2, c3):
    """Admissibility W(Dx) > 0 for all Dx > 0  <=>  c3 > (2/9) c2^2  (SI-2, Eq. S14).

    Only binding for bistable springs; every monotonic or snap-through spring
    (c2^2 <= 4 c3) satisfies it automatically.
    """
    return c3 > (2.0 / 9.0) * c2 ** 2


def spring_type(c2, c3):
    """Element-wise SI-3 spring type code (see SPRING_TYPES) of coefficient arrays."""
    s = c2 ** 2
    return np.where(~positive_energy(c2, c3), 3,
                    np.where(s > 4 * c3, 2, np.where(s > 3 * c3, 1, 0)))


def _sample_layers(rng, n, exclude_bistable):
    """n admissible (c2, c3) pairs, uniform over the admissible region (rejection sampling)."""
    out2 = np.empty(n)
    out3 = np.empty(n)
    filled = 0
    while filled < n:
        m = int((n - filled) * 1.6) + 16
        c2 = rng.uniform(*C2_RANGE, m)
        c3 = rng.uniform(*C3_RANGE, m)
        ok = positive_energy(c2, c3)
        if exclude_bistable:
            ok &= c2 ** 2 <= 4 * c3
        c2, c3 = c2[ok], c3[ok]
        k = min(len(c2), n - filled)
        out2[filled:filled + k] = c2[:k]
        out3[filled:filled + k] = c3[:k]
        filled += k
    return out2, out3


def sample_design_space(rng, n, category):
    """n random designs of a category, each (4,) for c2 and c3.

    homo        : one admissible (c2, c3) pair copied to all four families
    hetero      : four independent admissible pairs
    hetero_excl : four independent pairs that are also non-bistable (c2^2 <= 4 c3)
    """
    if category == "homo":
        c2, c3 = _sample_layers(rng, n, False)
        return np.repeat(c2[:, None], 4, axis=1), np.repeat(c3[:, None], 4, axis=1)
    c2, c3 = _sample_layers(rng, 4 * n, category == "hetero_excl")
    return c2.reshape(n, 4), c3.reshape(n, 4)


def perturb(rng, c2, c3, k, eps, homogeneous):
    """k perturbed copies of each design (rows of c2/c3): paper Eq. 4 with resampling.

    c_i = c_i* (1 + delta_i),  delta_i ~ N(0, sigma^2),  sigma = eps / 2

    so ~95 % of copies have |delta| <= eps. c2 and c3 get independent deltas. One delta
    per spring family (all 5 springs of a family move together, so a heterogeneous
    chain keeps four distinct spring types); with ``homogeneous=True`` one delta for the
    whole chain, so it stays homogeneous (paper Sec. 2.2). Copies violating the
    positive-energy constraint are redrawn.

    Returns (c2, c3) of shape (n * k, 4); rows i*k .. i*k + k-1 belong to design i.
    """
    sigma = eps / 2.0
    n = c2.shape[0]
    P2 = np.repeat(c2, k, axis=0)
    P3 = np.repeat(c3, k, axis=0)
    out2 = P2.copy()
    out3 = P3.copy()
    todo = np.arange(n * k)
    for _ in range(1000):
        if len(todo) == 0:
            break
        shape = (len(todo), 1) if homogeneous else (len(todo), 4)
        d2 = rng.normal(0.0, sigma, shape)
        d3 = rng.normal(0.0, sigma, shape)
        new2 = P2[todo] * (1 + d2)
        new3 = P3[todo] * (1 + d3)
        ok = positive_energy(new2, new3).all(axis=1)
        out2[todo[ok]] = new2[ok]
        out3[todo[ok]] = new3[ok]
        todo = todo[~ok]
    if len(todo):
        raise RuntimeError(f"{len(todo)} perturbations could not be made admissible")
    return out2, out3


def build_inputs(args, rng):
    """Returns c2, c3 (rows, 4) plus per-row design_id, is_nominal, category code.

    With --category mixed the designs are split into equal thirds (homo, hetero,
    hetero_excl). With --perturb K each design occupies K + 1 consecutive rows:
    the nominal design first, then its K perturbed copies.
    """
    rows = args.rows
    if args.sampler == "local":
        c2, c3 = sample_coeffs(rng, rows, args.spread)
        return c2, c3, np.arange(rows), np.ones(rows, bool), np.full(rows, -1)

    per_design = args.perturb + 1
    n_designs = rows // per_design
    if n_designs * per_design != rows:
        print(f"note: rows rounded down to {n_designs * per_design} ({n_designs} designs x {per_design})")
    cats = list(CATEGORIES) if args.category == "mixed" else [args.category]
    split = np.array_split(np.arange(n_designs), len(cats))

    C2, C3, DID, NOM, CAT = [], [], [], [], []
    for cat, ids in zip(cats, split):
        d2, d3 = sample_design_space(rng, len(ids), cat)
        if args.perturb:
            p2, p3 = perturb(rng, d2, d3, args.perturb, args.eps, cat == "homo")
            # block layout per design: [nominal, K perturbed]
            d2b = np.concatenate([d2[:, None, :], p2.reshape(len(ids), args.perturb, 4)], axis=1)
            d3b = np.concatenate([d3[:, None, :], p3.reshape(len(ids), args.perturb, 4)], axis=1)
            d2, d3 = d2b.reshape(-1, 4), d3b.reshape(-1, 4)
        C2.append(d2)
        C3.append(d3)
        DID.append(np.repeat(ids, per_design))
        nom = np.zeros(per_design, bool)
        nom[0] = True
        NOM.append(np.tile(nom, len(ids)))
        CAT.append(np.full(len(ids) * per_design, CATEGORIES[cat]))
    return (np.concatenate(C2), np.concatenate(C3), np.concatenate(DID),
            np.concatenate(NOM), np.concatenate(CAT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=360_000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--sampler", choices=["paper", "local"], default="paper")
    ap.add_argument("--category", choices=["mixed", "hetero", "hetero_excl", "homo"], default="mixed")
    ap.add_argument("--perturb", type=int, default=0, help="perturbed copies per nominal design")
    ap.add_argument("--eps", type=float, default=0.05, help="perturbation noise level (paper: 0-0.05)")
    ap.add_argument("--spread", type=float, default=0.3, help="only for --sampler local")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="gpu = CUDA kernel in fast_dem_cuda.py (needs numba-cuda + NVIDIA GPU)")
    args = ap.parse_args()

    if args.device == "gpu":
        from numba import cuda
        try:
            import fast_dem_cuda as G
        except ImportError:
            raise SystemExit("--device gpu needs fast_dem_cuda.py, which is not in this "
                             "repository; use --device cpu")
        if not cuda.is_available():
            raise SystemExit("No CUDA GPU visible (check nvidia-smi and that numba-cuda is installed)")
        name = getattr(cuda.get_current_device(), "name", "simulator")
        print(f"GPU: {name.decode() if isinstance(name, bytes) else name}")
        solve = G.batch_max_ke_gpu
        if args.chunk == 20_000:
            args.chunk = 120_000  # bigger launches keep the GPU full
    else:
        if args.threads:
            numba.set_num_threads(args.threads)
        print(f"threads: {numba.get_num_threads()}")
        solve = F.batch_max_ke

    rng = np.random.default_rng(args.seed)
    c2_all, c3_all, design_id, is_nominal, category = build_inputs(args, rng)
    rows = len(c2_all)
    types = spring_type(c2_all, c3_all)
    print(f"{rows} rows; layer spring types: "
          + ", ".join(f"{SPRING_TYPES[t]} {100 * (types == t).mean():.1f}%" for t in range(4)))

    ratio = np.full(rows, np.nan)
    steps = np.zeros(rows, dtype=np.int64)
    status = np.zeros(rows, dtype=np.int64)

    # compile + linear reference (once). The ratio denominator is shared by every row.
    lin = F.linear_max_ke()
    solve(c2_all[:2], c3_all[:2], args.rtol, args.atol)

    # checkpoint after every chunk so a long run can be inspected / salvaged
    tmp = args.out + ".partial.npz"
    t0 = time.time()
    for s in range(0, rows, args.chunk):
        e = min(s + args.chunk, rows)
        ke, st, ss = solve(c2_all[s:e], c3_all[s:e], args.rtol, args.atol)
        r = ke / lin
        r[ss != 0] = np.nan
        ratio[s:e], steps[s:e], status[s:e] = r, st, ss
        el = time.time() - t0
        print(f"{e:>8}/{rows} rows  {el:7.1f}s  {e / el:7.0f} rows/s  "
              f"ETA {(rows - e) * el / e:6.0f}s  failed so far {(status[:e] != 0).sum()}",
              flush=True)
        np.savez(tmp, c2=c2_all[:e], c3=c3_all[:e], ratio=ratio[:e], steps=steps[:e], status=status[:e])

    np.savez(args.out, c2=c2_all, c3=c3_all, ratio=ratio, steps=steps, status=status,
             design_id=design_id, is_nominal=is_nominal, category=category, spring_type=types,
             linear_max_ke=lin, rtol=args.rtol, atol=args.atol, sampler=args.sampler,
             perturb=args.perturb, eps=args.eps, spread=args.spread, seed=args.seed)
    os.remove(tmp)
    print(f"done: {rows} rows in {time.time() - t0:.1f}s -> {args.out}")
    print(f"steps per row: median {np.median(steps):.0f}, max {steps.max()}; failed rows: {(status != 0).sum()}")


if __name__ == "__main__":
    main()