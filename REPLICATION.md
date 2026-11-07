# Replication status

Status of reproducing Tamur et al. (main text + SI) with this code. All numbers below
can be regenerated with `fast/check_simulation.py` (simulation checks) and
`fast/good.py` (robustness).

## 1. The simulator is verified

| Check | Result |
|---|---|
| Equations of motion, fast_dem vs MATLAB, 5000 random states (all branches: tension/compression, contact on/off, past the bistable well) | max diff 6e-16 (scaled) |
| Equations of motion, scipy port vs MATLAB | max rel. diff 1e-14 |
| KE ratio, fast_dem vs MATLAB ode45, 16 designs | median rel. diff 6e-7, max 2e-5 |
| Paper Table S2, 4 of 6 published designs | within 0.4–3 % (paper prints 2 significant figures) |
| Paper Fig. 4 (impactor mass 1.01 M): linear peak-KE increase | ×1.018 (paper ×1.02) |
| Paper Fig. 4: KE ratio homo / hetero / hetero-excl | 0.109 / 0.026 / 0.0077 (paper 11 % / 2.6 % / 0.8 %) |

The model matches paper Eq. 5 / SI Eq. S1.14: Hertzian impactor contact with no tension,
springs linear in tension and cubic in compression, 2ζ dampers on every chain spring
including the wall spring, no damper on the contact.

## 2. Known discrepancies, and why they are not simulator bugs

**Two published designs don't reproduce Table S2:** homogeneous optimum (sim 0.0231 vs
paper 0.0117) and heterogeneous perturbed (sim 0.0123 vs 0.036). Both contain bistable
springs. If c2/c3 are moved by less than ±0.005 (the rounding of the printed table), the
KE ratio sweeps 0.0117–0.062 and 0.0117–0.038 respectively. Both ranges contain the
paper values exactly. The designs without this sensitivity agree to < 3 %. Conclusion:
Table S1 is printed with too few digits for bistable designs; MATLAB, scipy and fast_dem
all agree with each other. To close this, ask the authors for full-precision
coefficients.

**Output grid.** The paper states Δt_out = 1/(100 f0). The MATLAB code instead uses
1/(100 f) with f from k_f = max(1, 1 + 2c2 + 3c3), which is ~50–100× finer. fast_dem
takes the peak continuously from the solver's dense output. On the six paper designs,
all three choices agree to < 0.3 %, so the choice doesn't matter for the results.

## 3. Status per figure

| Paper item | Status | What it needs |
|---|---|---|
| Single-design KE ratio (objective J) | ✅ done, verified | – |
| Fig. 2 (KE over impactor M–V space) | 🟡 simulator ready (`m_imp`, `v_imp` arguments, verified at 1.01 M) | sweep script + plotting |
| Fig. 1c (KE vs noise level ε, 500 samples) | 🟡 perturbation model ready (`generate_dataset.perturb`, Eq. 4) | loop over ε for the three paper designs + plot |
| Fig. 1d / SI-4 (category robustness) | 🟡 pipeline runs (`good.py`) on random designs | optimised baseline sets (see below) |
| Fig. 3, 4, S3, S4 (spatiotemporal KE/PE/stiffness, FFT, pulse FWHM) | 🔴 not started | full trajectory output (fast_dem only returns the peak KE); `scipy_ref` can provide trajectories |
| Inverse design (fmincon, Methods 5.2) | 🔴 not started (deliberately postponed) | optimiser over c2/c3 with the SI-2/SI-3 constraints |

### Preliminary Fig. 1d from random designs (not yet the paper protocol)

`python good.py --data dataset.npz --k 20` (eps 5 %, KE window [0.01, 0.03]):

| category | baseline designs | median μ_KE | P(μ_KE < 0.03) | paper median / P |
|---|---|---|---|---|
| homogeneous | 20 | 0.104 | 0 % | 0.105 / 0 % |
| heterogeneous | 84 | 0.042 | 23 % | 0.067 / 0 % |
| het. excl. bistability | 60 | 0.032 | 40 % | 0.050 / 9 % |

The category ordering matches the paper (homo worst, het.-excl. best), and the
homogeneous value matches almost exactly. The heterogeneous sets come out more robust
than in the paper, which is expected: these baselines are random samples that happen to
land in the window, while the paper's are optimised designs sitting in sharp minima.
Too few random designs reach the window (the paper uses 200 per category), so the
inverse-design step is needed to replicate Fig. 1d properly.
