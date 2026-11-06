"""
Compare fast_dem against the original scipy implementation on random perturbed rows.

    python validate.py --ref-dir ../ref --n 16

--ref-dir must contain the original (scipy) main_disorder_v2.py, impact_equation_of_motion_hetero.py
and impact_force.py. Each reference row takes ~6 s, so rows run in parallel processes.
"""
import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import fast_dem as F
from generate_dataset import sample_coeffs

_REF_FN = None


def _load_reference(ref_dir):
    """Load the scipy main_disorder_v2 from ref_dir by file path, so a torch version
    with the same module names elsewhere on sys.path can't be picked up by mistake."""
    ref_dir = os.path.abspath(ref_dir)
    path = os.path.join(ref_dir, "main_disorder_v2.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no main_disorder_v2.py in {ref_dir}")
    # the reference imports these two by name; make sure they come from ref_dir
    for name in ("impact_equation_of_motion_hetero", "impact_force"):
        sys.modules.pop(name, None)
    sys.path.insert(0, ref_dir)
    spec = importlib.util.spec_from_file_location("ref_main_disorder_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if "torch" in sys.modules:
        raise RuntimeError(f"{ref_dir} looks like the torch version, not the scipy reference")
    return mod.main_disorder_v2


def _ref(args):
    global _REF_FN
    ref_dir, c2, c3 = args
    if _REF_FN is None:
        _REF_FN = _load_reference(ref_dir)
    return _REF_FN(c2, c3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--spread", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    c2, c3 = sample_coeffs(rng, a.n, a.spread)
    fast = F.main_disorder_v2(c2, c3)
    with ProcessPoolExecutor(a.workers) as ex:
        ref = np.array(list(ex.map(_ref, [(a.ref_dir, c2[i], c3[i]) for i in range(a.n)])))
    rel = np.abs(fast / ref - 1)
    for i in range(a.n):
        print(f"{i:3d}  ref {ref[i]:.8f}  fast {fast[i]:.8f}  rel.err {rel[i]:.1e}")
    print(f"max rel. error {rel.max():.2e}, median {np.median(rel):.2e}")


if __name__ == "__main__":
    main()