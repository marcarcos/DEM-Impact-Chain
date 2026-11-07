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

## DEM and Gaussian process notebook

[Spring_GP_Colab.ipynb](Spring_GP_Colab.ipynb) provides a self-contained Python
workflow for DEM simulation and Gaussian process (GP) optimization of the spring
coefficients. It includes constrained Sobol initialization, GP-guided search,
regional refinement, optional one-at-a-time sensitivity sweeps, candidate checks
with RK45 and DOP853, and result visualizations. It uses its own SciPy solver;
no MATLAB installation or repository dataset is required for a fresh search.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcarcos/DEM-Impact-Chain/blob/main/Spring_GP_Colab.ipynb)

1. Open the notebook in Google Colab, or open it locally in Jupyter with a Python 3 kernel.
2. Review the settings in **Section 1** before running the remaining cells in order.
   The setup cell installs missing dependencies: NumPy, pandas, SciPy,
   scikit-learn, Matplotlib, Numba, joblib, and threadpoolctl.
3. Start with the supplied `RUN_MODE = "small_test"`. This runs a reduced search
   for the heterogeneous category with fewer designs and rounds; it still
   performs fresh DEM solves and is not a full reproduction of the experiment.
4. Set `RUN_MODE = "new_search"` for the full configured search across homogeneous,
   heterogeneous, and heterogeneous excluding bistability designs. This performs
   substantially more simulations and GP fits, so allow additional runtime.

| Setting | Default | Effect |
|---|---|---|
| `RUN_SENSITIVITY` | `False` | Enable one-at-a-time coefficient perturbations around the notebook baseline. These sweeps do not enforce design feasibility. |
| `RECHECK_SELECTED_DESIGNS` | `True` | Recompute selected candidates with both RK45 and DOP853. |
| `RUN_PREDICTION_TEST` | `False` | Enable an independent local prediction test using additional DEM evaluations. |
| `USE_GOOGLE_DRIVE` | `False` | Mount Google Drive and save results there when running in Colab. |
| `DOWNLOAD_RESULTS` | `True` | Download the result ZIP automatically at the end of a Colab run. |

The optional `RUN_MODE = "saved_presentation"` requires a separate historical
results archive, which is not bundled with this notebook. Before using that mode,
define `PRESENTATION_DATA = Path("/path/to/extracted/archive")` before Section 10.
The archive must contain `results/`, `IMPROVED_CANDIDATES.json`, and the regional
logs under `improvement/multiregion_gp/` and `improvement/gp_resume/` expected by
`load_presentation_records` in Section 8.

Results are saved under `/content/Week6_DEM_GP/<RUN_MODE>/` in Colab, or
`Week6_DEM_GP/<RUN_MODE>/` relative to the working directory locally. With Drive
enabled, the root is `/content/drive/MyDrive/Week6_DEM_GP/<RUN_MODE>/`.
Outputs include CSV and JSON evaluation records, search checkpoints, serialized
GP models, PNG/SVG figures, final design summaries, and provenance/validation
manifests. The final cell creates `Week6_GP_Results_<RUN_MODE>.zip` alongside the
run directory. Keep the same output directory and configuration to reuse logged
evaluations and available checkpoints; use a new directory when changing settings.

The uploaded notebook contains no executed cell outputs. Historical reference
values are distinct from fresh search results, and a new search does not guarantee
the same optimum. The notebook does not execute native MATLAB validation; see
[REPLICATION.md](REPLICATION.md) for the repository's simulator validation.

## Layout

| Path | What it is |
|---|---|
| `Spring_GP_Colab.ipynb` | Self-contained Colab/Jupyter DEM and GP optimization workflow; run modes, sensitivity analysis, candidate verification, and exports. |
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
