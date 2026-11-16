"""Space-time matrices of the chain response: Python version of the plots in
matlab/main_disorder.m (time on the y axis, mass index on the x axis, colour = value).

Runs the same two solves as main_disorder.m, with the same solver (RK45 = ode45's
Dormand-Prince 5(4)), tolerances (1e-12) and output grid (0:dt:T), so the matrices
match MATLAB's X_lin / X_non entry for entry up to solver round-off.

    python plot_matrices.py                                     # MATLAB example design
    python plot_matrices.py --c2 -84.36 -94.08 -10.11 -33.35 --c3 1589.77 2426.41 1682.13 3300.14
    python plot_matrices.py --quantity displacement --tmax 150 --out het_disp.png

Matrices, as in MATLAB (rows = time steps):
    velocity_x_*      X[:, 1::2]  (N+1 columns: column 0 = impactor, 1..N = chain)
    displacement_x_*  X[:, 2::2]  (N columns: chain masses 1..N)
Both are saved to <out>.npz together with t, so they can be compared with MATLAB.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from impact_equation_of_motion_hetero import impact_equation_of_motion_hetero


def simulate(c2, c3, mass_factor=1.0):
    """Linear and nonlinear solves exactly as in main_disorder.m. Returns t, X_lin, X_non.

    mass_factor scales the impactor mass (1.01 = the perturbed case of the paper's Fig. 4).
    """
    c1 = np.ones(4)
    c2 = np.asarray(c2, dtype=float)
    c3 = np.asarray(c3, dtype=float)
    C1, C2, C3 = np.repeat(c1, 5), np.repeat(c2, 5), np.repeat(c3, 5)

    zeta = 0.02
    N = 20
    m_layer = 1.0
    M0_impactor = N / 2
    C_impact = 1.2e4
    M_impactor = 0.01 * M0_impactor * mass_factor
    V_impactor = 0.25

    k_f = max(1.0, np.max(1 + 2 * c2 + 3 * c3))
    cycles, outputpercycle = 25, 100
    f0 = 1 / (2 * np.pi) * np.sqrt(1 / M0_impactor)
    dt_cyc0 = 1 / f0
    f = 1 / (2 * np.pi) * np.sqrt(k_f / M0_impactor)
    dt = (1 / f) / outputpercycle
    T = dt_cyc0 * cycles
    t_eval = np.arange(int(np.floor(T / dt)) + 1) * dt   # MATLAB's 0:dt:T

    material_info = np.array([m_layer, M_impactor, C_impact, zeta])
    x0 = np.zeros(2 * (N + 1))
    x0[1] = V_impactor
    opts = dict(method="RK45", t_eval=t_eval, rtol=1e-12, atol=np.full(2 * (N + 1), 1e-12))

    lin_info = np.column_stack((C1, np.zeros(N), np.zeros(N)))
    non_info = np.column_stack((C1, C2, C3))
    X_lin = solve_ivp(impact_equation_of_motion_hetero, (0, t_eval[-1]), x0,
                      args=(lin_info, material_info, 0.0), **opts).y.T
    X_non = solve_ivp(impact_equation_of_motion_hetero, (0, t_eval[-1]), x0,
                      args=(non_info, material_info, 0.0), **opts).y.T
    return t_eval, X_lin, X_non


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c2", type=float, nargs=4, default=[-70, -100, -5, -35])
    ap.add_argument("--c3", type=float, nargs=4, default=[1200, 2000, 1500, 3000])
    ap.add_argument("--quantity", choices=["velocity", "displacement", "ke"], default="velocity")
    ap.add_argument("--impactor", action="store_true", help="include the impactor column (velocity/KE only)")
    ap.add_argument("--tmax", type=float, default=None, help="only plot t <= tmax (default: whole run)")
    ap.add_argument("--out", default="matrices.png")
    a = ap.parse_args()

    t, X_lin, X_non = simulate(a.c2, a.c3)

    # same slicing as MATLAB: X(:,2:2:2*(N+1)) and X(:,3:2:2*(N+1))
    velocity_x_lin, velocity_x_non = X_lin[:, 1::2], X_non[:, 1::2]
    displacement_x_lin, displacement_x_non = X_lin[:, 2::2], X_non[:, 2::2]
    N = displacement_x_lin.shape[1]

    KE_lin = 0.5 * velocity_x_lin[:, N] ** 2
    KE_non = 0.5 * velocity_x_non[:, N] ** 2
    print(f"Maximum KE ratio at the last cell = {KE_non.max() / KE_lin.max():.6f}")
    np.savez_compressed(a.out.rsplit(".", 1)[0] + ".npz", t=t,
                        velocity_x_lin=velocity_x_lin, velocity_x_non=velocity_x_non,
                        displacement_x_lin=displacement_x_lin, displacement_x_non=displacement_x_non)

    first = 0 if a.impactor else 1   # column of the first plotted mass in the velocity matrix
    if a.quantity == "velocity":
        M_lin, M_non = velocity_x_lin[:, first:], velocity_x_non[:, first:]
        label, cmap = "velocity", "RdBu_r"
    elif a.quantity == "ke":
        M_lin, M_non = 0.5 * velocity_x_lin[:, first:] ** 2, 0.5 * velocity_x_non[:, first:] ** 2
        label, cmap = "kinetic energy", "magma"
    else:
        M_lin, M_non, first = displacement_x_lin, displacement_x_non, 1
        label, cmap = "displacement", "RdBu_r"

    keep = slice(None) if a.tmax is None else t <= a.tmax
    t_plot, M_lin, M_non = t[keep], M_lin[keep], M_non[keep]

    # one colour scale for both panels so they can be compared directly
    vmax = max(np.abs(M_lin).max(), np.abs(M_non).max())
    vmin = 0.0 if a.quantity == "ke" else -vmax
    x_lo, x_hi = first - 0.5, first + M_lin.shape[1] - 0.5
    extent = [x_lo, x_hi, t_plot[0], t_plot[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True, constrained_layout=True)
    for ax, M, name in [(axes[0], M_lin, "Linear chain"), (axes[1], M_non, "Nonlinear chain")]:
        im = ax.imshow(M, aspect="auto", origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("mass index (0 = impactor, 1 = impacted end, 20 = next to wall)"
                      if first == 0 else "mass index (1 = impacted end, 20 = next to wall)")
        ax.set_xticks(([0] if first == 0 else []) + [1, 5, 10, 15, 20])
        for b in (5.5, 10.5, 15.5):   # spring-family boundaries A|B, B|C, C|D
            ax.axvline(b, color="0.5", lw=0.6, ls=":")
        for x, fam in zip((3, 8, 13, 18), "ABCD"):   # family labels above the panel
            ax.text(x, 1.005, fam, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=9, color="0.35")
    axes[0].set_ylabel("time t")
    fig.colorbar(im, ax=axes, label=label, shrink=0.9)
    fig.suptitle(f"c2 = {list(a.c2)}, c3 = {list(a.c3)}", fontsize=10)
    fig.savefig(a.out, dpi=200)
    print(f"saved {a.out} and {a.out.rsplit('.', 1)[0]}.npz")


if __name__ == "__main__":
    main()
