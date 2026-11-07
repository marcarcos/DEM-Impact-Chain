# DEM Impact Chain

Discrete-element simulation of an impacted 1D chain with heterogeneous nonlinear springs,
replicating Tamur, Brandy, Kim & Boechler, *Heterogeneity in constitutive nonlinearity
enhances impact mitigation robustness* (main text + SI).

A rigid impactor (mass M, velocity V) hits a chain of N = 20 unit masses joined by
springs that are linear in tension and cubic in compression,
`f(Δx) = Δx + c2 Δx² + c3 Δx³`, with linear dampers (ζ = 0.02) and a fixed wall at the far end.
The 20 springs form four families A–D of 5 springs, so a design is `c2, c3 ∈ R⁴`.
The performance metric is the **KE ratio** = peak kinetic energy of the last mass,
nonlinear chain / linear chain (lower is better).

## Quick start

```bash
pip install numpy numba scipy matplotlib
```

```bash
cd fast && python check_simulation.py
```

`check_simulation.py` verifies the simulator against MATLAB and the paper in seconds
(see [REPLICATION.md](REPLICATION.md)). To compute KE ratios in Python:

```python
import fast_dem as F
F.main_disorder_v2([-84.36, -94.08, -10.11, -33.35], [1589.77, 2426.41, 1682.13, 3300.14])  # 0.00569
F.main_disorder_v2(c2_batch, c3_batch)            # (B, 4) arrays -> (B,) ratios, all cores
F.main_disorder_v2(c2, c3, m_imp=1.01 * F.M_IMPACTOR, v_imp=0.25)  # other impact condition
```

## Layout

| Path | What it is |
|---|---|
| `matlab/` | Original MATLAB model (reference; not modified). `main_disorder.m` example, `main_disorder_fn.m` function form, `impact_equation_of_motion_hetero.m` EOM, `run_validation.m` / `rhs_check.m` produce the stored `matlab_results.csv` / `rhs_matlab.csv`. |
| `scipy_ref/` | Line-by-line scipy port of the MATLAB code (slow, ~6 s/design). Used to arbitrate between MATLAB and the fast code. `sim_data.py` is an exploratory one-at-a-time sensitivity sweep. |
| `fast/fast_dem.py` | **Production simulator.** Numba-compiled Dormand–Prince 5(4) (same as scipy RK45 / MATLAB ode45), batched over CPU threads, continuous peak-KE detection. ~500–650 designs/s on 4 cores. |
| `fast/check_simulation.py` | One-command pass/fail check vs MATLAB and the paper. |
| `fast/rhs_check.py` | Equations of motion: fast vs scipy vs MATLAB on 5000 random states. |
| `fast/validate_all.py` | KE ratio: fast vs scipy vs MATLAB vs paper on 16 designs. |
| `fast/validate.py` | KE ratio: fast vs scipy on random designs near the MATLAB example. |
| `fast/generate_dataset.py` | Random designs from the paper's design space (per category, optional Monte Carlo perturbations) → `.npz`. |
| `fast/good.py` | Paper Fig. 1d protocol: balanced baseline in a KE window + 5 % Monte Carlo design uncertainty. |
| `fast/dataset.npz` | 360k random nominal designs (equal thirds homo / hetero / hetero-excl), from `generate_dataset.py --seed 0`. |
| `fast/data_exp.ipynb` | Quick exploration of `dataset.npz`. |

## Conventions

* Everything is non-dimensional (SI-1): m = c1 = a = 1, V0 = 1 (linear sound speed),
  M0 = N/2 = 10. Design point M/M0 = 0.01, V/V0 = 0.25, C_imp = 1.2e4, T = 25/f0 ≈ 496.7.
* Families A→D run from the impacted end to the wall.
* Chain categories: `homo` (one spring law), `hetero` (four), `hetero_excl` (four, none
  bistable: c2² ≤ 4c3). All designs must satisfy positive strain energy c3 > (2/9)c2².
* Spring types (SI-3): monotonic c2² ≤ 3c3, snap-through 3c3 < c2² ≤ 4c3, bistable c2² > 4c3.
